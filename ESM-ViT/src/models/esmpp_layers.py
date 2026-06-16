"""
ESM++ (Essential Subspace Merging ++) — Dynamic per-sample routing layers.

Per-sample routing: each sample independently selects its best-matching LoRA expert
via cosine similarity to task prototypes. Uses fully parallel bmm to avoid
sequential expert loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple


class LoRAExpert(nn.Module):
    """Low-rank expert parameters for a single task at a single layer.

    B @ A ≈ ΔW (task vector) low-rank approximation.
    - A: (r, d_in)  — projection coefficients
    - B: (d_out, r) — principal directions
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.register_buffer('A', A.clone())  # (r, d_in)
        self.register_buffer('B', B.clone())  # (d_out, r)

    def get_delta(self) -> torch.Tensor:
        """Returns the low-rank weight delta: B @ A, shape (d_out, d_in)."""
        return self.B @ self.A


class MoERoutedLinearPerSample(nn.Module):
    """Per-sample routed replacement for nn.Linear (mlp.c_fc).

    Routes each sample independently via cosine similarity to prototypes,
    then applies per-sample effective weights in a single parallel bmm.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        experts: Dict[str, LoRAExpert],
        prototypes: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Base weights from main model (frozen)
        self.base_weight = nn.Parameter(base_weight.clone())
        self.base_weight.requires_grad_(False)
        if base_bias is not None:
            self.base_bias = nn.Parameter(base_bias.clone())
            self.base_bias.requires_grad_(False)
        else:
            self.register_parameter('base_bias', None)

        # Experts: ModuleList indexed by position
        self._task_names: List[str] = list(experts.keys())
        self.experts = nn.ModuleList(
            [experts[name] for name in self._task_names]
        )

        # Prototypes: pre-normalized and stacked for vectorized routing
        proto_list = [prototypes[name] for name in self._task_names]
        self.register_buffer(
            'prototypes', F.normalize(torch.stack(proto_list, dim=0), dim=-1),
            persistent=False,
        )

        # Pre-compute all deltas stacked (T, d_out, d_in) on CPU; moved to GPU on use
        self._num_experts = len(self._task_names)
        self.forced_task: Optional[str] = None

        # Recording state for routing accuracy measurement
        self.recording: bool = False
        self._recorded_indices: List[torch.Tensor] = []

    def set_forced_expert(self, task_name: Optional[str]):
        assert task_name is None or task_name in self._task_names, \
            f"Unknown task '{task_name}'. Available: {self._task_names}"
        self.forced_task = task_name

    def start_recording(self):
        """Clear previous recordings and enable routing index collection."""
        self._recorded_indices = []
        self.recording = True

    def stop_recording(self):
        """Disable routing index collection."""
        self.recording = False

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Return concatenated recorded expert indices, or None if empty."""
        if self._recorded_indices:
            return torch.cat(self._recorded_indices, dim=0)
        return None

    def _get_all_deltas_cpu(self) -> torch.Tensor:
        """Return stacked deltas (T, d_out, d_in) on CPU, cached after first call.

        Kept on CPU to avoid OOM: for ViT-L-14 × 20 tasks, all cached GPU deltas
        would consume ~14 GB across 48 routed layers.
        """
        if not hasattr(self, '_cached_deltas_cpu'):
            self._cached_deltas_cpu = torch.stack(
                [e.get_delta() for e in self.experts]
            )
        return self._cached_deltas_cpu

    def _route_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample routing: returns (B,) LongTensor of expert indices."""
        B = x.shape[0]

        if self.forced_task is not None:
            idx = self._task_names.index(self.forced_task)
            return torch.full((B,), idx, device=x.device, dtype=torch.long)

        # Per-sample feature: mean over tokens (if 3D) or raw (if 2D)
        if x.dim() == 3:
            x_feat = x.mean(dim=1)  # (B, d_in)
        else:
            x_feat = x  # (B, d_in)

        # Cosine similarity with all prototypes: (B, d_in) @ (d_in, T) = (B, T)
        x_norm = F.normalize(x_feat, dim=-1)
        sims = torch.mm(x_norm, self.prototypes.t())
        indices = sims.argmax(dim=-1)  # (B,) — one expert index per sample

        if self.recording:
            self._recorded_indices.append(indices.detach().cpu())

        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = self._route_per_sample(x)  # (B,)

        # Index into CPU deltas; only the selected (B, d_out, d_in) subset
        # moves to GPU — avoids materializing all T deltas on GPU.
        all_deltas_cpu = self._get_all_deltas_cpu()  # (T, d_out, d_in) on CPU
        selected_deltas = all_deltas_cpu[indices.cpu()].to(
            device=x.device, dtype=x.dtype
        )  # (B, d_out, d_in)

        # Base output + per-sample delta contributions (no W_eff materialization)
        base_out = F.linear(x, self.base_weight, self.base_bias)

        if x.dim() == 3:
            # (B, N, d_in) @ (B, d_in, d_out) -> (B, N, d_out)
            delta_out = torch.bmm(x, selected_deltas.transpose(1, 2))
        else:
            # (B, d_in) -> (B, 1, d_in) @ (B, d_in, d_out) -> (B, 1, d_out) -> (B, d_out)
            delta_out = torch.bmm(x.unsqueeze(1), selected_deltas.transpose(1, 2)).squeeze(1)

        return base_out + delta_out


class MoEMultiheadAttentionPerSample(nn.Module):
    """Per-sample routed replacement for nn.MultiheadAttention.

    Routes each sample independently, then applies per-sample in_proj weights
    via parallel bmm. The attention computation itself remains batched.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        in_proj_weight: torch.Tensor,
        in_proj_bias: Optional[torch.Tensor],
        out_proj_weight: torch.Tensor,
        out_proj_bias: Optional[torch.Tensor],
        experts: Dict[str, LoRAExpert],
        prototypes: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        # Base in-proj weights (frozen)
        self.register_parameter('in_proj_weight', nn.Parameter(in_proj_weight.clone()))
        self.in_proj_weight.requires_grad_(False)
        if in_proj_bias is not None:
            self.register_parameter('in_proj_bias', nn.Parameter(in_proj_bias.clone()))
            self.in_proj_bias.requires_grad_(False)
        else:
            self.register_parameter('in_proj_bias', None)

        # Output projection (frozen)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.weight.data.copy_(out_proj_weight.data)
        self.out_proj.weight.requires_grad_(False)
        if out_proj_bias is not None:
            self.out_proj.bias.data.copy_(out_proj_bias.data)
            self.out_proj.bias.requires_grad_(False)
        else:
            self.out_proj.bias = None

        # Experts (indexed)
        self._task_names: List[str] = list(experts.keys())
        self.experts = nn.ModuleList(
            [experts[name] for name in self._task_names]
        )

        # Stacked prototypes (T, embed_dim), pre-normalized
        proto_list = [prototypes[name] for name in self._task_names]
        self.register_buffer(
            'prototypes', F.normalize(torch.stack(proto_list, dim=0), dim=-1),
            persistent=False,
        )

        self.forced_task: Optional[str] = None

        # Recording state for routing accuracy measurement
        self.recording: bool = False
        self._recorded_indices: List[torch.Tensor] = []

    def set_forced_expert(self, task_name: Optional[str]):
        assert task_name is None or task_name in self._task_names, \
            f"Unknown task '{task_name}'. Available: {self._task_names}"
        self.forced_task = task_name

    def start_recording(self):
        """Clear previous recordings and enable routing index collection."""
        self._recorded_indices = []
        self.recording = True

    def stop_recording(self):
        """Disable routing index collection."""
        self.recording = False

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Return concatenated recorded expert indices, or None if empty."""
        if self._recorded_indices:
            return torch.cat(self._recorded_indices, dim=0)
        return None

    def _get_all_deltas_cpu(self) -> torch.Tensor:
        """Return stacked deltas (T, 3*embed_dim, embed_dim) on CPU, cached after
        first call.

        Kept on CPU to avoid OOM: for ViT-L-14 × 20 tasks, all cached GPU deltas
        would consume ~14 GB across 48 routed layers.
        """
        if not hasattr(self, '_cached_deltas_cpu'):
            self._cached_deltas_cpu = torch.stack(
                [e.get_delta() for e in self.experts]
            )
        return self._cached_deltas_cpu

    def _route_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample routing: returns (B,) LongTensor of expert indices."""
        B = x.shape[0]

        if self.forced_task is not None:
            idx = self._task_names.index(self.forced_task)
            return torch.full((B,), idx, device=x.device, dtype=torch.long)

        # Per-sample feature: mean over tokens
        if x.dim() == 3:
            x_feat = x.mean(dim=1)  # (B, embed_dim)
        else:
            x_feat = x  # (B, embed_dim)

        x_norm = F.normalize(x_feat, dim=-1)
        sims = torch.mm(x_norm, self.prototypes.t())
        indices = sims.argmax(dim=-1)

        if self.recording:
            self._recorded_indices.append(indices.detach().cpu())

        return indices

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention defaults
        if key is None:
            key = query
        if value is None:
            value = key

        # === Per-sample MoE routing on in_proj ===
        # query: (B, L, E) — OpenCLIP convention (batch, seq, embed)
        B, L, E_ = query.shape

        indices = self._route_per_sample(query)  # (B,)

        # Index into CPU deltas; only selected (B, 3*E, E) subset moves to GPU.
        all_deltas_cpu = self._get_all_deltas_cpu()  # (T, 3*E, E) on CPU
        selected_deltas = all_deltas_cpu[indices.cpu()].to(
            device=query.device, dtype=query.dtype
        )  # (B, 3*E, E)

        # === Per-sample QKV projection via bmm ===
        # Split base + delta to avoid materializing W_eff (B, 3*E, E).
        base_qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        delta_qkv = torch.bmm(query, selected_deltas.transpose(1, 2))
        qkv = base_qkv + delta_qkv

        q, k, v = qkv.chunk(3, dim=-1)
        # q, k, v: (B, L, E) each

        # === Multi-head attention (batched, same as original) ===
        head_dim = self.head_dim

        q = q.reshape(B, L * self.num_heads, head_dim).transpose(0, 1)
        k = k.reshape(B, L * self.num_heads, head_dim).transpose(0, 1)
        v = v.reshape(B, L * self.num_heads, head_dim).transpose(0, 1)
        # (L * num_heads, B, head_dim) = (seq * num_heads, batch, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        attn_output = attn_output.transpose(0, 1).reshape(B, L, E_)

        # === Output projection (unchanged, shared across experts) ===
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, None
        return attn_output, None


def _get_parent_module(model: nn.Module, name: str):
    """Given a module name, returns (parent_module, child_name)."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_moe_model_per_sample(
    model: nn.Module,
    routed_experts: Dict[str, Dict[str, LoRAExpert]],
    routed_prototypes: Dict[str, Dict[str, torch.Tensor]],
) -> nn.Module:
    """Build ESM++ model by replacing target layers with per-sample routing variants.

    Replaces attention and MLP layers with per-sample routing variants
    (:class:`MoERoutedLinearPerSample` /
    :class:`MoEMultiheadAttentionPerSample`).
    """
    for name, module in list(model.named_modules()):
        # === Replace nn.MultiheadAttention with MoEMultiheadAttentionPerSample ===
        if name.endswith('.attn'):
            layer_key = name + '.in_proj_weight'
            if layer_key not in routed_experts:
                continue

            print(f"Replacing {name} with MoEMultiheadAttentionPerSample")
            parent, child_name = _get_parent_module(model, name)

            new_attn = MoEMultiheadAttentionPerSample(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                in_proj_weight=module.in_proj_weight.data,
                in_proj_bias=module.in_proj_bias.data if module.in_proj_bias is not None else None,
                out_proj_weight=module.out_proj.weight.data,
                out_proj_bias=module.out_proj.bias.data if module.out_proj.bias is not None else None,
                experts=routed_experts[layer_key],
                prototypes=routed_prototypes[layer_key],
            )
            setattr(parent, child_name, new_attn)

        # === Replace nn.Linear(c_fc) with MoERoutedLinearPerSample ===
        if name.endswith('.c_fc'):
            layer_key = name + '.weight'
            if layer_key not in routed_experts:
                continue

            print(f"Replacing {name} with MoERoutedLinearPerSample")
            parent, child_name = _get_parent_module(model, name)

            new_linear = MoERoutedLinearPerSample(
                in_features=module.in_features,
                out_features=module.out_features,
                base_weight=module.weight.data,
                base_bias=module.bias.data if module.bias is not None else None,
                experts=routed_experts[layer_key],
                prototypes=routed_prototypes[layer_key],
            )
            setattr(parent, child_name, new_linear)

    replaced = len(routed_experts)
    print(f"Per-sample MoE model built. Replaced {replaced} layer groups.")
    return model


def set_forced_expert_per_sample(model: nn.Module, task_name: Optional[str]):
    """Set or clear the forced expert on every per-sample routed layer."""
    for module in model.modules():
        if isinstance(module, (MoERoutedLinearPerSample, MoEMultiheadAttentionPerSample)):
            module.set_forced_expert(task_name)


def set_recording_per_sample(model: nn.Module, recording: bool = True):
    """Enable or disable routing index recording on all per-sample MoE layers.

    When enabled, each layer records the expert index selected for each sample
    during :meth:`_route_per_sample`.  Retrieve with
    :func:`get_all_routing_decisions_per_sample`.
    """
    for module in model.modules():
        if isinstance(module, (MoERoutedLinearPerSample, MoEMultiheadAttentionPerSample)):
            if recording:
                module.start_recording()
            else:
                module.stop_recording()


def get_all_routing_decisions_per_sample(model: nn.Module) -> Dict[str, Dict]:
    """Collect recorded routing decisions from all per-sample MoE layers.

    Returns:
        Dict mapping module name to ``{"indices": LongTensor, "task_names": list[str]}``.
        Layers with no recorded data are omitted.
    """
    decisions = {}
    for name, module in model.named_modules():
        if isinstance(module, (MoERoutedLinearPerSample, MoEMultiheadAttentionPerSample)):
            indices = module.get_recorded_indices()
            if indices is not None:
                decisions[name] = {
                    "indices": indices,
                    "task_names": module._task_names,
                }
    return decisions
