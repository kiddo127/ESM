import torch
import tqdm
import utils
from collections import defaultdict
from param import param


class MergingMethod:

    @utils.args_inspector
    def __init__(
        self,
        models_to_merge,
        models_name,
    ):
        self.models_name = {n: i for i, n in enumerate(models_name)}
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    # ==================================================================
    #  ESM (Essential Subspace Merging)
    # ==================================================================
    @utils.args_inspector
    @torch.inference_mode()
    def esm(
        self,
        base_model,
        models_to_merge_dict: dict,
        scaling: float = 1.0,
        principal_direction_dict=None,
    ):
        T = len(models_to_merge_dict)
        sv_reduction = 1.0 / T
        orthogonalize = True
        detect_task_name = list(models_to_merge_dict.keys())[0]

        print(f"ESM (merging): fusing {T} tasks, sv_reduction={sv_reduction:.3f}")

        task_vectors = [model - base_model for model in models_to_merge_dict.values()]
        task_vectors_dict = {task: model - base_model for task, model in models_to_merge_dict.items()}

        merged_param_dict = {}
        param_names = list(base_model.param_dict.keys())
        self.merged_task_vector_norm = 0

        for param_name in tqdm.tqdm(param_names, desc="ESM processing params"):
            base_param = base_model.param_dict[param_name]

            if len(base_param.shape) == 2:
                if param_name in principal_direction_dict[detect_task_name]:
                    merged_param = self._esm_process_2d_param(
                        param_name, base_param, task_vectors_dict, T,
                        sv_reduction, principal_direction_dict, orthogonalize, scaling,
                    )
                else:
                    merged_param = self._tsv_process_2d_param(
                        param_name, base_param, task_vectors, T,
                        sv_reduction, orthogonalize, scaling,
                    )
            else:
                merged_param = self._tsv_process_1d_param(
                    param_name, base_param, task_vectors, scaling,
                )

            merged_param_dict[param_name] = merged_param

        print('merged task vector norm: ', self.merged_task_vector_norm)
        self.merged_task_vector_norm = 0

        merged_param_dict = self.inter_layer_scaling(base_model.param_dict, merged_param_dict)
        merged_param = param(merged_param_dict)
        return merged_param

    # ==================================================================
    #  ESM++ (Essential Subspace Routing — no weight fusion)
    # ==================================================================
    @utils.args_inspector
    @torch.inference_mode()
    def esm_pp(
        self,
        base_model,
        models_to_merge_dict: dict,
        scaling: float = 1.0,
        principal_direction_dict=None,
        orthogonalize_v: bool = True,
        rank: int = 0,
    ):
        """
        Extract per-task (LoRA-style) experts for every weight matrix.
        Returns expert_dict for dynamic routing, not fused weights.

        Args:
            rank: low-rank expert dimension; 0 means auto d/T
        Returns:
            expert_dict: {param_name: {task_name: {'U':, 'V':, 'V_router':, 'norm':}}}
            base_model_params: base_model state_dict
        """
        T = len(models_to_merge_dict)
        sv_reduction = 1.0 / T if rank <= 0 else 0
        detect_task_name = list(models_to_merge_dict.keys())[0]

        task_vectors_dict = {task: model - base_model for task, model in models_to_merge_dict.items()}

        expert_dict = {}
        param_names = list(base_model.param_dict.keys())

        for param_name in tqdm.tqdm(param_names, desc="Extract MoE experts"):
            base_param = base_model.param_dict[param_name]

            if len(base_param.shape) == 2 and param_name in principal_direction_dict.get(detect_task_name, {}):
                experts = self._extract_experts_for_moe(
                    param_name, base_param, task_vectors_dict,
                    principal_direction_dict, sv_reduction,
                    orthogonalize_v=orthogonalize_v, rank=rank,
                )
                expert_dict[param_name] = experts

        return expert_dict, base_model.param_dict

    @utils.args_inspector
    @torch.inference_mode()
    def collect_prototypes_for_moe(
        self,
        base_model,
        finetuned_models,
        val_data_path="data/validation.json",
        proxy_num=32,
        device="cuda:0",
    ):
        """Collect prototype embeddings for prototype-based MoE routing."""
        from esm_moe_eval import collect_prototypes

        target_layer_names = []
        for name, _ in base_model.named_modules():
            if (name.endswith(".attention.self.query") or
                name.endswith(".attention.self.key") or
                name.endswith(".attention.self.value") or
                name.endswith(".attention.output.dense") or
                name.endswith(".intermediate.dense") or
                name.endswith(".output.dense")):
                target_layer_names.append(name)

        return collect_prototypes(
            target_layer_names=target_layer_names,
            finetuned_models=finetuned_models,
            val_data_path=val_data_path,
            proxy_num=proxy_num,
            device=device,
        )

    # ==================================================================
    #  Internal helpers
    # ==================================================================
    def _esm_process_2d_param(self, param_name, base_param, task_vectors_dict, T,
                               sv_reduction, principal_direction_dict, orthogonalize, scaling):
        device = base_param.device
        task_matrices = {}
        for task_name, tv in task_vectors_dict.items():
            task_matrices[task_name] = tv.param_dict[param_name]

        U_list, V_list = [], []

        for task_name, task_matrix in task_matrices.items():
            principal_directions = principal_direction_dict[task_name][param_name]['eigenvectors'].cuda()
            U = principal_directions
            V = principal_directions.T @ task_matrix

            k = max(1, int(principal_directions.shape[1] * sv_reduction))
            U_reduced = U[:, :k]
            V_reduced = V[:k, :]
            eigenvalues = principal_direction_dict[task_name][param_name]['eigenvalues'].cuda()

            # eigenvalue-based scaling
            eigenvalues_reduced = eigenvalues[:k]
            eigenvalue_scale = torch.pow(eigenvalues_reduced / eigenvalues_reduced.mean(), 0.15)
            eigenvalue_scale = eigenvalue_scale / eigenvalue_scale.mean()
            U_reduced = U_reduced * eigenvalue_scale.unsqueeze(0)
            V_reduced = V_reduced * eigenvalue_scale.unsqueeze(1)

            U_list.append(U_reduced)
            V_list.append(V_reduced)

        U_concatenated = torch.cat(U_list, dim=1)
        V_concatenated = torch.cat(V_list, dim=0)

        if orthogonalize:
            U_ortho = self._procrustes_orthogonalize(U_concatenated)
            V_ortho = self._procrustes_orthogonalize(V_concatenated)
        else:
            U_ortho, V_ortho = U_concatenated, V_concatenated

        T_optimal = U_ortho @ V_ortho
        final_param = base_param + scaling * T_optimal
        self.merged_task_vector_norm += torch.norm(scaling * T_optimal).item()
        return final_param

    def _tsv_process_2d_param(self, param_name, base_param, task_vectors, T,
                               sv_reduction, orthogonalize, scaling):
        device = base_param.device
        task_matrices = []
        for tv in task_vectors:
            task_matrices.append(tv.param_dict[param_name])

        U_list, S_list, V_list = [], [], []
        for task_matrix in task_matrices:
            U, S, V = torch.linalg.svd(task_matrix, full_matrices=False)
            k = max(1, int(S.shape[0] * sv_reduction))
            U_list.append(U[:, :k])
            S_list.append(S[:k])
            V_list.append(V[:k, :])

        U_concatenated = torch.cat(U_list, dim=1)
        S_concatenated = torch.cat(S_list, dim=0)
        V_concatenated = torch.cat(V_list, dim=0)

        if orthogonalize:
            U_ortho = self._procrustes_orthogonalize(U_concatenated)
            V_ortho = self._procrustes_orthogonalize(V_concatenated)
        else:
            U_ortho, V_ortho = U_concatenated, V_concatenated

        S_diag = torch.diag(S_concatenated)
        merged_matrix = U_ortho @ S_diag @ V_ortho
        final_param = base_param + scaling * merged_matrix
        self.merged_task_vector_norm += torch.norm(scaling * merged_matrix).item()
        return final_param

    def _tsv_process_1d_param(self, param_name, base_param, task_vectors, scaling):
        task_params = []
        for tv in task_vectors:
            task_params.append(tv.param_dict[param_name])
        avg_task_param = torch.stack(task_params).mean(dim=0)
        return base_param + scaling * avg_task_param

    def _procrustes_orthogonalize(self, X):
        """Procrustes orthogonalization: find closest orthogonal matrix to X."""
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        X_ortho = U @ Vh * S.mean()
        return X_ortho

    def _extract_experts_for_moe(self, param_name, base_param, task_vectors_dict,
                                  principal_direction_dict, sv_reduction,
                                  orthogonalize_v=True, rank=0):
        """
        Extract per-task MoE experts (U_i, V_i, V_router_i) for a single param.
        Orthgonalized V serves as router; raw U, V as expert parameters.
        """
        device = base_param.device
        experts = {}
        task_names = list(task_vectors_dict.keys())

        for task_name, tv in task_vectors_dict.items():
            task_matrix = tv.param_dict[param_name]
            principal_directions = principal_direction_dict[task_name][param_name]['eigenvectors'].to(device)

            U = principal_directions
            V = U.T @ task_matrix

            if rank > 0:
                k = min(principal_directions.shape[1], rank)
            else:
                k = max(1, int(principal_directions.shape[1] * sv_reduction))
            U_reduced = U[:, :k].contiguous()
            V_reduced = V[:k, :].contiguous()

            experts[task_name] = {
                'U': U_reduced.half(),
                'V': V_reduced.half(),
            }

        # Task-orthogonalize V for routing
        if orthogonalize_v and len(task_names) > 1:
            V_cat = torch.cat([experts[t]['V'].float() for t in task_names], dim=0)
            V_cat_ortho = self._procrustes_orthogonalize(V_cat)

            for i, task_name in enumerate(task_names):
                k_i = experts[task_name]['V'].shape[0]
                start = sum(experts[t]['V'].shape[0] for t in task_names[:i])
                experts[task_name]['V_router'] = V_cat_ortho[start:start + k_i].half()
        else:
            for task_name in task_names:
                experts[task_name]['V_router'] = experts[task_name]['V'].clone()

        # Compute norms from V_router
        for task_name in task_names:
            experts[task_name]['norm'] = torch.norm(experts[task_name]['V_router']).item()

        return experts

    @torch.inference_mode()
    def inter_layer_scaling(self, base_param_dict, merged_param_dict, exponent=2):
        """Inter-layer scale normalization."""
        norms = {
            'attention.self.query.weight': [],
            'attention.self.key.weight': [],
            'attention.self.value.weight': [],
            'attention.output.dense.weight': [],
            'intermediate.dense.weight': [],
            'output.dense.weight': [],
        }
        for key in merged_param_dict.keys():
            for norms_key in norms.keys():
                if norms_key in key:
                    task_vector = merged_param_dict[key] - base_param_dict[key]
                    norms[norms_key].append(torch.norm(task_vector))
                    break
        for key in norms.keys():
            norms[key] = torch.tensor(norms[key]).cuda().mean()
        for key in merged_param_dict.keys():
            for norms_key in norms.keys():
                if norms_key in key:
                    task_vector = merged_param_dict[key] - base_param_dict[key]
                    scale = torch.norm(task_vector) / norms[norms_key]
                    merged_param_dict[key] = base_param_dict[key] + task_vector * pow(scale, exponent)
                    break
        return merged_param_dict
