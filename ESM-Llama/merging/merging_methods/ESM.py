"""
Essential Subspace Merging (ESM).

For every linear layer that has recorded data-driven principal directions, each
task's weight delta  dW = W_ft - W_base  is expressed in that task's own
output-offset principal basis:

    U = principal directions (eigenvectors of the output-offset covariance)
    V = U^T @ dW

Then, per layer:
  1. keep the top k = d_out * sv_reduction components per task,
  2. weight U by an eigenvalue factor (lambda / mean(lambda))**eig_exp,
  3. concatenate U and V across tasks,
  4. Procrustes-orthogonalize U (only),
  5. reconstruct the merged task vector  T = Procrustes(U) @ V,
  6. W_merged = W_base + scaling * T.

Only U is orthogonalized: V is kept raw so its per-task / per-direction
magnitudes are preserved.  Orthogonalizing V, or re-weighting V by the
eigenvalues, destroys that magnitude structure and collapses performance.

Layers without recorded principal directions (e.g. lm_head) fall back to TSV
(SVD truncation + Procrustes orthogonalization); 1D parameters (norms, biases)
are averaged.
"""
import torch
from tqdm import tqdm
from collections import OrderedDict
from merging_methods.utils import select_trainable_params
from merging_methods.merger import Merger


class ESM(Merger):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)
        self.merged_task_vector_norm = 0

    def _procrustes_orthogonalize(self, X):
        """Closest orthogonal matrix to X (scaled by the mean singular value)."""
        U, S, Vh = torch.linalg.svd(X.to(torch.float32), full_matrices=False)
        return (U @ Vh * S.mean()).to(X.dtype)

    def _load_layer_stats(self, path):
        """Load layer_stats.pt -> {task: {layer_name: {eigenvectors, eigenvalues}}}."""
        layer_stats = torch.load(path, map_location='cpu', weights_only=True)
        principal_direction_dict = {}
        for task_name, task_layers in layer_stats.items():
            principal_direction_dict[task_name] = {
                layer_name: {
                    'eigenvectors': info['offset_eigenvectors'],
                    'eigenvalues': info['offset_eigenvalues'],
                }
                for layer_name, info in task_layers.items()
            }
        return principal_direction_dict

    def _esm_process_2d_param(self, param_name, base_param, task_vectors_dict, T,
                              sv_reduction, principal_direction_dict, scaling, eig_exp,
                              use_eig_weight=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_p_gpu = base_param.to(device).float()

        U_list, V_list = [], []
        for task_name, ft_sd in task_vectors_dict.items():
            task_matrix = ft_sd[param_name].to(device).float() - base_p_gpu
            eigenvectors = principal_direction_dict[task_name][param_name]['eigenvectors'].to(device).float()
            eigenvalues = principal_direction_dict[task_name][param_name]['eigenvalues'].to(device).float()

            # Project dW into the task's output principal basis
            V_all = eigenvectors.T @ task_matrix
            k = max(1, int(eigenvectors.shape[1] * sv_reduction))

            U_k = eigenvectors[:, :k]
            if use_eig_weight:
                # Eigenvalue weighting (applied to U only)
                ev = eigenvalues[:k]
                eig_scale = torch.pow(ev / (ev.mean() + 1e-12), eig_exp)
                eig_scale = eig_scale / (eig_scale.mean() + 1e-12)
                U_k = U_k * eig_scale.unsqueeze(0)

            U_list.append(U_k)
            V_list.append(V_all[:k, :])                      # V kept raw
            del task_matrix, eigenvectors, eigenvalues, V_all

        U_concat = torch.cat(U_list, dim=1)
        V_concat = torch.cat(V_list, dim=0)
        del U_list, V_list

        U_ortho = self._procrustes_orthogonalize(U_concat)
        T_optimal = U_ortho @ V_concat
        del U_concat, V_concat, U_ortho

        result = (base_p_gpu + scaling * T_optimal).to(base_param.dtype).cpu()
        self.merged_task_vector_norm += torch.norm(scaling * T_optimal).item()
        del base_p_gpu, T_optimal
        return result

    def _tsv_process_2d_param(self, param_name, base_param, ft_state_dicts, T,
                              sv_reduction, scaling):
        """Fallback for 2D params without principal directions: SVD + Procrustes."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_p_gpu = base_param.to(device).float()

        U_list, S_list, Vh_list = [], [], []
        for i in range(T):
            task_matrix = ft_state_dicts[i][param_name].to(device).float() - base_p_gpu
            U, S, Vh = torch.linalg.svd(task_matrix, full_matrices=False)
            k = max(1, int(S.shape[0] * sv_reduction))
            U_list.append(U[:, :k].clone())
            S_list.append(S[:k].clone())
            Vh_list.append(Vh[:k, :].clone())
            del task_matrix, U, S, Vh

        U_concat = torch.cat(U_list, dim=1)
        S_concat = torch.cat(S_list, dim=0)
        Vh_concat = torch.cat(Vh_list, dim=0)
        del U_list, S_list, Vh_list

        U_ortho = self._procrustes_orthogonalize(U_concat)
        Vh_ortho = self._procrustes_orthogonalize(Vh_concat.T).T
        merged_task = U_ortho @ torch.diag(S_concat) @ Vh_ortho
        del U_concat, Vh_concat, U_ortho, Vh_ortho, S_concat

        result = (base_p_gpu + scaling * merged_task).to(base_param.dtype).cpu()
        self.merged_task_vector_norm += torch.norm(scaling * merged_task).item()
        del base_p_gpu, merged_task
        return result

    def _tsv_process_1d_param(self, param_name, base_param, ft_state_dicts, scaling):
        """Average the task deltas for 1D parameters."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_p_gpu = base_param.to(device).float()

        sum_delta = torch.zeros_like(base_p_gpu)
        for ft_sd in ft_state_dicts:
            sum_delta += ft_sd[param_name].to(device).float() - base_p_gpu

        final_p = base_p_gpu + scaling * (sum_delta / len(ft_state_dicts))
        result = final_p.to(base_param.dtype).cpu()
        del base_p_gpu, sum_delta, final_p
        return result

    @torch.inference_mode()
    def merge(self, **kwargs):
        scaling = kwargs.get("scaling_coef", 0.9)
        T = len(self.ft_ckpts)
        sv_reduction = kwargs.get("sv_reduction", 0.1)
        eig_exp = kwargs.get("eig_exp", 0.15)
        use_eig_weight = kwargs.get("use_eig_weight", True)
        layer_stats_path = kwargs.get("layer_stats_path", "./layer_stats.pt")

        print(f"ESM: merging {T} tasks, sv_reduction={sv_reduction:.3f}, scaling={scaling}, "
              f"eig_exp={eig_exp}, use_eig_weight={use_eig_weight}")
        principal_direction_dict = self._load_layer_stats(layer_stats_path)
        task_names = list(principal_direction_dict.keys())
        detect_task_name = task_names[0]
        print(f"  Layers with principal directions: {len(principal_direction_dict[detect_task_name])}")

        base_sd = select_trainable_params(self.base_model)
        ft_sds = [select_trainable_params(m) for m in self.ft_ckpts]
        task_vectors_dict = OrderedDict((t, ft_sds[i]) for i, t in enumerate(task_names))

        self.merged_task_vector_norm = 0
        merged_params = {}
        for param_name in tqdm(base_sd.keys(), desc="ESM Merging"):
            base_param = base_sd[param_name]
            if len(base_param.shape) == 2:
                if param_name in principal_direction_dict.get(detect_task_name, {}):
                    merged_params[param_name] = self._esm_process_2d_param(
                        param_name, base_param, task_vectors_dict, T,
                        sv_reduction, principal_direction_dict, scaling, eig_exp,
                        use_eig_weight=use_eig_weight)
                else:
                    # un-recorded 2D params (e.g. lm_head): scaled by 1.0
                    merged_params[param_name] = self._tsv_process_2d_param(
                        param_name, base_param, ft_sds, T, sv_reduction, 1.0)
            else:
                merged_params[param_name] = self._tsv_process_1d_param(
                    param_name, base_param, ft_sds, 1.0)

        print(f"Merged task vector norm: {self.merged_task_vector_norm:.4f}")
        self.base_model.load_state_dict(merged_params, strict=False)
        self.base_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
