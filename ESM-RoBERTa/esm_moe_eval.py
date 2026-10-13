import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import tqdm
from collections import defaultdict


DATA_ROOT = os.environ.get('DATA_ROOT', 'data')


def _stack_prototypes_rt(prototypes, task_names):
    """Stack per-task prototypes into [sum_k, d] + a center->task-index map."""
    rows, center = [], []
    for i, t in enumerate(task_names):
        p = prototypes[t]
        if p.dim() == 1:
            p = p.unsqueeze(0)
        rows.append(p.float())
        center += [i] * p.shape[0]
    return F.normalize(torch.cat(rows, dim=0), dim=-1), torch.tensor(center, dtype=torch.long)


class MoERoutedLinear(nn.Module):
    """Per-sample routed drop-in replacement for nn.Linear (RoBERTa).

    Routing = the same mean-pool -> cosine -> argmax as the evaluator, but
    packaged as a real module so the whole model can be traced by torch.compile
    into ONE graph per shape (instead of one specialized graph per layer).
    Expert application is the (mathematically identical) `F.linear(x, W_base+dW)`
    — the fused weight is selected directly, no weight swap.
    """

    def __init__(self, base_weight, base_bias, experts, prototypes, factored=False):
        super().__init__()
        dev, dt = base_weight.device, base_weight.dtype
        self.factored = factored
        self.register_buffer('base_weight', base_weight.clone(), persistent=False)
        if base_bias is not None:
            self.register_buffer('base_bias', base_bias.clone(), persistent=False)
        else:
            self.register_buffer('base_bias', None, persistent=False)
        self._task_names = list(experts.keys())
        if factored:
            # Keep the low-rank factors (shared across samples) and apply
            # x@(U@V)^T == (x@V^T)@U^T — avoids materializing B dense (d_out,d_in)
            # per-sample weights, which is the large-batch memory bottleneck.
            U_list, V_list = [], []
            for t in self._task_names:
                U_list.append(experts[t]['U'].to(device=dev, dtype=dt))   # (d_out, r)
                V_list.append(experts[t]['V'].to(device=dev, dtype=dt))   # (r, d_in)
            self.register_buffer('U_stack', torch.stack(U_list, 0), persistent=False)  # (T,d_out,r)
            self.register_buffer('V_stack', torch.stack(V_list, 0), persistent=False)  # (T,r,d_in)
        else:
            fused = []
            for t in self._task_names:
                U = experts[t]['U'].to(device=dev, dtype=dt)
                V = experts[t]['V'].to(device=dev, dtype=dt)
                fused.append(base_weight + U @ V)
            self.register_buffer('fused', torch.stack(fused, dim=0), persistent=False)  # (T,d_out,d_in)
        proto, center = _stack_prototypes_rt(prototypes, self._task_names)
        self.register_buffer('prototypes', proto.to(dev), persistent=False)
        self.register_buffer('_center_to_task', center.to(dev), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.mean(dim=1) if x.dim() == 3 else x
        sims = F.normalize(xf, dim=-1) @ self.prototypes.t()      # (B, sum_k)
        idx = self._center_to_task[sims.argmax(dim=-1)]           # (B,) expert index
        if self.factored:
            base_out = F.linear(x, self.base_weight, self.base_bias)
            U = self.U_stack[idx]                                 # (B, d_out, r)
            V = self.V_stack[idx]                                 # (B, r, d_in)
            if x.dim() == 3:
                z = torch.bmm(x, V.transpose(1, 2))               # (B, S, r)
                return base_out + torch.bmm(z, U.transpose(1, 2))
            z = torch.bmm(x.unsqueeze(1), V.transpose(1, 2))      # (B, 1, r)
            return base_out + torch.bmm(z, U.transpose(1, 2)).squeeze(1)
        w = self.fused[idx]                                       # (B, d_out, d_in)
        if idx.numel() == 1:
            # Single sample: use plain F.linear with the selected 2-D weight so
            # the math is identical to the evaluator's weight-swap path
            # (routing is near-tie; bmm vs linear rounding can flip decisions).
            return F.linear(x, w[0], self.base_bias)
        if x.dim() == 3:
            out = torch.bmm(x, w.transpose(1, 2))                 # (B, S, d_out)
        else:
            out = torch.bmm(x.unsqueeze(1), w.transpose(1, 2)).squeeze(1)
        if self.base_bias is not None:
            out = out + self.base_bias
        return out


def build_routed_model(model: nn.Module, expert_dict, prototype_dict, factored=False) -> nn.Module:
    """Replace every routed nn.Linear with MoERoutedLinear (in place)."""
    n = 0
    for param_name, experts in expert_dict.items():
        module_name = param_name[:-len('.weight')]
        parent = model
        parts = module_name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        setattr(parent, parts[-1], MoERoutedLinear(
            old.weight.data, old.bias.data if old.bias is not None else None,
            experts, prototype_dict[param_name], factored=factored))
        n += 1
    print(f"Replaced {n} layers with MoERoutedLinear")
    return model


# ------------------------------------------------------------------
#  Prototype collection utilities
# ------------------------------------------------------------------
def _masked_prototype_hook(layer_weight_name, collected, collected_masks, current_mask_ref):
    """Hook that captures layer input features alongside per-token validity mask."""
    def hook(module, input_tensor, output_tensor):
        x = input_tensor[0].detach().cpu()
        collected.setdefault(layer_weight_name, []).append(x)
        mask = current_mask_ref[0]
        if mask is not None:
            collected_masks.setdefault(layer_weight_name, []).append(mask.cpu())
        else:
            seq_len = x.shape[1]
            collected_masks.setdefault(layer_weight_name, []).append(
                torch.ones(x.shape[0], seq_len)
            )
    return hook


@torch.inference_mode()
def _kmeans(X, k, iters=25, seed=0):
    """Minimal Lloyd's k-means. X:[N,d] -> centers [k,d]."""
    N, d = X.shape
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:k].to(X.device)
    C = X[idx].clone()
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = X[m].mean(dim=0)
    return C


def collect_prototypes(
    target_layer_names,
    finetuned_models,
    val_data_path=os.path.join(DATA_ROOT, 'validation.json'),
    proxy_num=64,
    device="cuda:0",
    pool="mean",
    n_clusters=1,
    shared_model=None,
):
    """
    Collect prototype embeddings for prototype-based MoE routing.

    For each task, runs the corresponding finetuned model on that task's proxy
    data and captures layer-wise input features. Padding tokens are masked out
    so prototypes only reflect non-padding positions, consistent with routing.
    """
    import utils
    import eval as eval_mod

    data = utils.from_json(val_data_path)

    task_data = defaultdict(list)
    for data_item in data:
        task_id = data_item['dataset_ids']
        task_data[task_id].append(data_item)

    proxy_data = {}
    for task_id, samples in task_data.items():
        if len(samples) <= proxy_num:
            proxy_data[task_id] = samples
        else:
            proxy_data[task_id] = random.sample(samples, proxy_num)

    prototype_dict = {}

    for data_id, samples in proxy_data.items():
        data_name = list(eval_mod.glue_data_id_map.keys())[data_id]
        model = shared_model if shared_model is not None else finetuned_models[data_name]
        # Per-task cluster count: n_clusters may be an int or a {task: k} dict.
        if isinstance(n_clusters, dict):
            k_t = n_clusters.get(data_name, 1)
        else:
            k_t = n_clusters if n_clusters else 1
        if isinstance(k_t, str) and k_t.isdigit():
            k_t = int(k_t)

        collected = {}
        collected_masks = {}
        current_mask_ref = [None]
        hook_handles = []

        for name, module in model.named_modules():
            if name in target_layer_names:
                weight_name = name + ".weight"
                handle = module.register_forward_hook(
                    _masked_prototype_hook(weight_name, collected, collected_masks, current_mask_ref)
                )
                hook_handles.append(handle)

        for data_item in tqdm.tqdm(samples, desc=f'Collect prototypes for {data_name}'):
            input_ids = torch.tensor(data_item['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(data_item['attention_mask']).unsqueeze(0).to(device)
            current_mask_ref[0] = attention_mask
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Build per-sample pooled features for each layer (needed for k>1)
        multi = isinstance(k_t, int) and k_t > 1
        Fs = {}
        if multi:
            for weight_name, feature_list in collected.items():
                masks = collected_masks[weight_name]
                feats = []
                for f, m in zip(feature_list, masks):
                    f = f.reshape(-1, f.shape[-1])          # [S, d]
                    if pool == "cls":
                        feats.append(f[0])                  # [CLS] token
                    else:
                        valid = m.reshape(-1) > 0
                        feats.append(f[valid].mean(dim=0))  # masked token mean
                Fs[weight_name] = torch.stack(feats, 0).float()

        k_use = int(k_t) if isinstance(k_t, int) else 1

        for weight_name, feature_list in collected.items():
            if multi:
                F = Fs[weight_name]
                if k_use > 1 and F.shape[0] >= k_use:
                    prototype = _kmeans(F, k_use)           # [k, d]
                else:
                    prototype = F.mean(dim=0, keepdim=True)
            elif pool == "cls":
                cls_feats = torch.cat([f[:, 0, :] for f in feature_list], dim=0)
                prototype = cls_feats.mean(dim=0, keepdim=True)
            else:
                all_features = torch.cat(
                    [f.reshape(-1, f.shape[-1]) for f in feature_list], dim=0
                )
                all_masks = torch.cat(
                    [m.reshape(-1) for m in collected_masks[weight_name]], dim=0
                )
                valid = all_masks > 0
                feature_sum = (all_features * valid.unsqueeze(1).float()).sum(dim=0)
                token_count = valid.sum().clamp(min=1)
                prototype = (feature_sum / token_count).unsqueeze(0)

            if weight_name not in prototype_dict:
                prototype_dict[weight_name] = {}
            prototype_dict[weight_name][data_name] = prototype

        for handle in hook_handles:
            handle.remove()

    return prototype_dict
