"""
Per-sample MoE routing for evaluation.

Unlike moe_model_for_eval.py which routes the entire batch to the same expert,
this module routes each sample independently. The key idea:

1. Routing phase: capture per-sample (not batch-pooled) activations via a forward
   pass, then compute cosine similarity against prototypes per sample.  With
   ``aggregate_routing`` the per-layer similarity scores are summed so each sample
   ends up with one consistent expert for every layer.

2. The resulting per-sample expert assignments are consumed by the eval drivers,
   which merge the selected experts into the weights (``fix_agg`` route-then-fix)
   before generation.

Usage in eval scripts::

    from moe_per_sample_router import PerSampleRouter

    router = PerSampleRouter(moe_model, aggregate_routing=True)
    router.compute_per_sample_routing(input_ids, attention_mask)
    dominant = router.get_dominant_expert_per_sample()
"""

import torch
from typing import Dict, Optional, List


class PerSampleRouter:
    """Per-sample routing for MoE models.

    Wraps an existing ``MoEModelForCausalLM`` instance and computes which expert
    each sample in a batch should use at every routed layer.  Weight merging and
    base-weight restoration are handled by the wrapped model.
    """

    def __init__(self, moe_model, aggregate_routing: bool = False):
        """
        Args:
            moe_model: MoEModelForCausalLM instance (already loaded).
            aggregate_routing: If True, aggregate cosine-similarity scores across
                all layers and assign each sample one consistent expert for every
                layer.  If False (default), each layer routes independently.
        """
        self._moe = moe_model
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.aggregate_routing = aggregate_routing

        # Per-sample routing results: layer_name -> LongTensor [batch_size]
        self._per_sample_routing: Dict[str, torch.Tensor] = {}

        # Routing cache for lm_eval loglikelihood (same-question optimisation)
        self._routing_cache: Dict[int, List[int]] = {}

        # Snapshot of routed-short-names so we don't chase moe_config on every call
        self._routed_short_names = list(moe_model.routed_short_names)
        self._num_layers = moe_model.base_model.config.num_hidden_layers

        # Prototypes per task: 1 (single mean) or k (k-means centroids).  With
        # k > 1 the prototype matrix is [num_experts * k, d_in] (task-major), so
        # the winning prototype index maps to a task via integer division by k.
        self._k = getattr(moe_model, 'prototype_k', 1)

        # Build ordered list of all routed layer names (used for cache serialisation)
        self._all_routed_layer_names: List[str] = []
        for layer_idx in range(self._num_layers):
            for sn in self._routed_short_names:
                self._all_routed_layer_names.append(
                    f'model.layers.{layer_idx}.{sn}.weight'
                )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def moe_model(self):
        return self._moe

    @property
    def per_sample_routing(self) -> Dict[str, torch.Tensor]:
        return self._per_sample_routing

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _seq_cache_key(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       seq_idx: int) -> int:
        """Hash the first 80 % of non-padding tokens of a single sequence."""
        valid_mask = attention_mask[seq_idx].bool()
        valid_tokens = input_ids[seq_idx][valid_mask]
        prefix_len = max(1, int(valid_tokens.shape[0] * 0.8))
        prefix = valid_tokens[:prefix_len]
        return hash(prefix.cpu().numpy().tobytes())

    def compute_per_sample_routing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        """Compute per-sample expert assignments for every routed layer.

        Steps
        -----
        1. If *use_cache* and every sample hits the cache, restore and return.
        2. Run one forward pass with hooks that capture per-sample mean
           activations (NOT batch-pooled).
        3. For each sample × layer: cosine-similarity against prototypes →
           best expert index.

        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq] (optional)
            use_cache: whether to consult / populate the routing cache.
        """
        routing_mode = self._moe.routing_mode

        # --- base / oracle: delegate to the original model (these don't need
        #     per-sample routing) ---
        if routing_mode in ('base', 'oracle'):
            self._moe.compute_routing_from_input(
                input_ids, attention_mask, use_cache=use_cache
            )
            self._per_sample_routing = {}
            return

        # --- auto mode: per-sample ---
        if attention_mask is None:
            pad_token_id = self._moe.base_model.config.pad_token_id
            if pad_token_id is None:
                pad_token_id = getattr(self._moe.base_model.config, 'eos_token_id', None) or 0
            attention_mask = (input_ids != pad_token_id).long()

        batch_size = input_ids.shape[0]

        # Cache lookup
        cache_keys = None
        if use_cache:
            cache_keys = tuple(
                self._seq_cache_key(input_ids, attention_mask, i)
                for i in range(batch_size)
            )
            if all(k in self._routing_cache for k in cache_keys):
                cached = self._routing_cache[cache_keys[0]]
                self._per_sample_routing = {}
                for i, name in enumerate(self._all_routed_layer_names):
                    expert_per_sample = [
                        self._routing_cache[ck][i] for ck in cache_keys
                    ]
                    self._per_sample_routing[name] = torch.tensor(
                        expert_per_sample, device=self._moe.device
                    )
                print(f"Per-sample routing: cached hit (batch={batch_size})")
                return

        # --- Capture per-sample activations ---
        moe_start = self._moe.moe_start_layer
        strategy = self._moe.routing_strategy

        # --- Single-pass routing with greedy LoRA ---
        # Combined hooks: capture input → route → apply LoRA, all in one hook.
        # This means when layer N fires, its input already includes the LoRA
        # corrections from layers moe_start .. N-1.  Deep layers route on the
        # true routed representations, not bare base-model activations.
        # After the pass, if aggregate_routing is set, per-layer similarity
        # scores are summed and the final routing is overridden so every layer
        # uses the same expert per sample.
        if strategy == 'per_block':
            # per_block uses input_layernorm output; keep the simpler two-round
            # approach for now (route on base, then refine with LoRA).
            per_sample_routing = self._capture_and_route_per_block(
                input_ids, attention_mask, moe_start, batch_size
            )
            per_sample_routing = self._refine_routing(
                input_ids, attention_mask, moe_start, batch_size,
                per_sample_routing, strategy,
            )
        elif self.aggregate_routing:
            per_sample_routing = self._single_pass_route_aggregated(
                input_ids, attention_mask, moe_start, batch_size
            )
        else:
            per_sample_routing = self._single_pass_route_per_layer(
                input_ids, attention_mask, moe_start, batch_size
            )

        self._per_sample_routing = per_sample_routing

        # Track routing accuracy
        self._track_routing_accuracy()

        # Print routing summary
        self._print_routing_summary(batch_size)

        # Populate cache
        if use_cache and cache_keys is not None:
            for bi, ck in enumerate(cache_keys):
                if ck not in self._routing_cache:
                    self._routing_cache[ck] = [
                        int(per_sample_routing[name][bi].item())
                        for name in self._all_routed_layer_names
                    ]

    def _capture_and_route_per_layer(
        self, input_ids, attention_mask, moe_start, batch_size
    ) -> Dict[str, torch.Tensor]:
        """per_layer strategy: each linear layer routes independently using its own input."""

        layer_inputs: Dict[str, torch.Tensor] = {}  # layer_name -> [batch, d_in]

        # Hoist per-call constants (bit-identical; mask fixed for the pass).
        mask_bool = attention_mask.bool()
        mask_unsq = mask_bool.unsqueeze(-1)
        counts = mask_bool.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [batch, 1]

        def make_hook(layer_idx: int, short_name: str):
            layer_name = f'model.layers.{layer_idx}.{short_name}.weight'

            def hook(module, args, output):
                x = args[0].detach()                     # [batch, seq, d_in]
                # Per-sample mean (vectorised over batch)
                x_sum = (x * mask_unsq).sum(dim=1)        # [batch, d_in]
                x_mean = x_sum / counts                   # [batch, d_in]
                layer_inputs[layer_name] = x_mean.float()

            return hook

        handles = []
        for layer_idx in range(moe_start, self._num_layers):
            for sn in self._routed_short_names:
                layer_module = self._moe.base_model.model.layers[layer_idx]
                module = layer_module
                for attr in sn.split('.'):
                    module = getattr(module, attr)
                h = module.register_forward_hook(make_hook(layer_idx, sn))
                handles.append(h)

        with torch.no_grad():
            _ = self._moe.base_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)

        for h in handles:
            h.remove()

        # Compute per-sample expert assignments (fully vectorised)
        routing: Dict[str, torch.Tensor] = {}
        for layer_name, x_mean in layer_inputs.items():
            proto = self._moe.prototype_matrix[layer_name]  # [num_experts, d_in]
            # cosine similarity: [batch, d_in] @ [d_in, num_experts] / norms
            x_norm = x_mean.norm(dim=1, keepdim=True)       # [batch, 1]
            p_norm = proto.norm(dim=1)                       # [num_experts]
            dots = x_mean @ proto.T                          # [batch, num_experts]
            sim = dots / (x_norm * p_norm.unsqueeze(0) + 1e-8)
            routing[layer_name] = sim.argmax(dim=1) // self._k  # [batch] task idx

        return routing

    def _capture_and_route_per_block(
        self, input_ids, attention_mask, moe_start, batch_size
    ) -> Dict[str, torch.Tensor]:
        """per_block strategy: all layers in a block share one routing decision."""

        block_inputs: Dict[int, torch.Tensor] = {}  # layer_idx -> [batch, d_model]

        # Hoist per-call constants (bit-identical; mask fixed for the pass).
        mask_bool = attention_mask.bool()
        mask_unsq = mask_bool.unsqueeze(-1)
        counts = mask_bool.sum(dim=1).clamp(min=1).unsqueeze(-1)

        def make_hook(layer_idx: int):
            def hook(module, args, output):
                x = output.detach()                          # post-layernorm [batch, seq, d_model]
                x_sum = (x * mask_unsq).sum(dim=1)           # [batch, d_model]
                x_mean = x_sum / counts
                block_inputs[layer_idx] = x_mean.float()

            return hook

        handles = []
        for layer_idx in range(moe_start, self._num_layers):
            ln = self._moe.base_model.model.layers[layer_idx].input_layernorm
            h = ln.register_forward_hook(make_hook(layer_idx))
            handles.append(h)

        with torch.no_grad():
            _ = self._moe.base_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)

        for h in handles:
            h.remove()

        # Route each block → broadcast to all routed layers in that block
        routing: Dict[str, torch.Tensor] = {}
        for layer_idx, x_mean in block_inputs.items():
            # Find a routing layer with compatible prototype dimension
            routing_layer = None
            for sn in self._routed_short_names:
                ln = f'model.layers.{layer_idx}.{sn}.weight'
                if ln in self._moe.prototype_matrix:
                    if self._moe.prototype_matrix[ln].shape[-1] == x_mean.shape[-1]:
                        routing_layer = ln
                        break
            if routing_layer is None:
                best = torch.zeros(batch_size, dtype=torch.long, device=self._moe.device)
            else:
                proto = self._moe.prototype_matrix[routing_layer]
                x_norm = x_mean.norm(dim=1, keepdim=True)
                p_norm = proto.norm(dim=1)
                dots = x_mean @ proto.T
                sim = dots / (x_norm * p_norm.unsqueeze(0) + 1e-8)
                best = sim.argmax(dim=1) // self._k

            for sn in self._routed_short_names:
                routing[f'model.layers.{layer_idx}.{sn}.weight'] = best

        return routing

    # ------------------------------------------------------------------
    # Two-round refinement (per_block only)
    # ------------------------------------------------------------------

    def _refine_routing(
        self, input_ids, attention_mask, moe_start, batch_size,
        initial_routing, strategy,
    ) -> Dict[str, torch.Tensor]:
        """Second forward pass with LoRA hooks active (per_block only).

        For per_block, the first pass uses base-model activations.  This
        refinement re-runs the capture with LoRA hooks so block-level routing
        sees the true routed representations.
        """
        prev_routing = self._per_sample_routing
        self._per_sample_routing = initial_routing
        self._register_lora_hooks()
        try:
            refined = self._capture_and_route_per_block(
                input_ids, attention_mask, moe_start, batch_size,
            )
        finally:
            self._remove_lora_hooks()
            self._per_sample_routing = prev_routing
        return refined

    # ------------------------------------------------------------------
    # Single-pass routing with greedy LoRA (combined hook)
    # ------------------------------------------------------------------

    def _single_pass_route_per_layer(
        self, input_ids, attention_mask, moe_start, _batch_size
    ) -> Dict[str, torch.Tensor]:
        """Single forward pass where each hook routes then immediately applies LoRA.

        Deep layers see LoRA-corrected inputs from shallower layers.  Each
        linear layer chooses its own expert independently (per_layer).
        """
        attention_mask_ref = [attention_mask]
        route_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._per_sample_routing = {}

        for layer_idx in range(moe_start, self._num_layers):
            for sn in self._routed_short_names:
                layer_name = f'model.layers.{layer_idx}.{sn}.weight'
                if layer_name not in self._moe.expert_lora:
                    continue
                self._register_combined_hook(
                    layer_name, attention_mask_ref, route_hooks,
                )

        with torch.no_grad():
            _ = self._moe.base_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)

        for h in route_hooks:
            h.remove()

        return dict(self._per_sample_routing)

    def _single_pass_route_aggregated(
        self, input_ids, attention_mask, moe_start, _batch_size
    ) -> Dict[str, torch.Tensor]:
        """Two-round aggregated routing with greedy-LoRA first pass.

        Round 1: combined hooks route + apply LoRA greedily per-layer (so deep
            layers see LoRA-corrected inputs).  Per-layer similarity vectors are
            collected and summed → first aggregated routing.

        Round 2 (refinement): LoRA hooks with the aggregated routing are
            registered, then a second forward pass re-captures activations.  Now
            ALL layers see the SAME expert's LoRA, so the similarity scores are
            based on the true routed-model representations.  Re-aggregate → final
            routing.

        Each sample ends up with ONE consistent expert for every layer.
        """
        # ---------- Round 1: greedy per-layer LoRA ----------
        attention_mask_ref = [attention_mask]
        route_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._per_sample_routing = {}

        per_layer_sim: Dict[str, torch.Tensor] = {}

        for layer_idx in range(moe_start, self._num_layers):
            for sn in self._routed_short_names:
                layer_name = f'model.layers.{layer_idx}.{sn}.weight'
                if layer_name not in self._moe.expert_lora:
                    continue
                self._register_combined_hook(
                    layer_name, attention_mask_ref, route_hooks,
                    collect_sim=per_layer_sim,
                )

        with torch.no_grad():
            _ = self._moe.base_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)

        for h in route_hooks:
            h.remove()

        if not per_layer_sim:
            return dict(self._per_sample_routing)

        # Aggregate round-1 scores
        num_experts = len(self._moe.prototype_matrix[
            list(per_layer_sim.keys())[0]
        ])
        acc_scores = torch.zeros(_batch_size, num_experts,
                                 device=self._moe.device, dtype=torch.float32)
        for sim in per_layer_sim.values():
            acc_scores += sim

        best_per_sample = acc_scores.argmax(dim=1) // self._k
        for layer_name in self._per_sample_routing:
            self._per_sample_routing[layer_name] = best_per_sample

        # ---------- Round 2: re-capture with aggregated LoRA ----------
        per_layer_sim_2: Dict[str, torch.Tensor] = {}
        capture_hooks: List[torch.utils.hooks.RemovableHandle] = []

        for layer_idx in range(moe_start, self._num_layers):
            for sn in self._routed_short_names:
                layer_name = f'model.layers.{layer_idx}.{sn}.weight'
                if layer_name not in self._moe.expert_lora:
                    continue
                h = self._register_capture_hook(
                    layer_name, attention_mask_ref, capture_hooks,
                    collect_sim=per_layer_sim_2,
                )

        self._register_lora_hooks()  # uses round-1 aggregated routing

        with torch.no_grad():
            _ = self._moe.base_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)

        for h in capture_hooks:
            h.remove()
        self._remove_lora_hooks()

        # Aggregate round-2 scores
        if per_layer_sim_2:
            acc_scores_2 = torch.zeros(_batch_size, num_experts,
                                       device=self._moe.device, dtype=torch.float32)
            for sim in per_layer_sim_2.values():
                acc_scores_2 += sim

            best_per_sample_2 = acc_scores_2.argmax(dim=1) // self._k
            for layer_name in self._per_sample_routing:
                self._per_sample_routing[layer_name] = best_per_sample_2

        return dict(self._per_sample_routing)

    def _print_routing_summary(self, batch_size: int):
        """Print per-sample routing distribution."""
        if not self._per_sample_routing:
            return
        from moe_config import TASK_NAMES

        idx_to_task = {i: t for i, t in enumerate(TASK_NAMES)}
        for bi in range(batch_size):
            counts = {t: 0 for t in TASK_NAMES}
            for experts in self._per_sample_routing.values():
                e = int(experts[bi].item())
                counts[idx_to_task[e]] += 1
            parts = ', '.join(f'{t}={c}' for t, c in counts.items())
            print(f"  Sample {bi}: {parts}")

    def _register_combined_hook(
        self,
        layer_name: str,
        attention_mask_ref: list,
        hook_list: List[torch.utils.hooks.RemovableHandle],
        collect_sim: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook that routes + applies LoRA in one shot.

        When the linear layer fires:
          1. capture per-sample mean of the input activation
          2. cosine-similarity vs prototypes -> expert per sample
          3. write expert decision into ``self._per_sample_routing``
          4. (optional) store similarity vector in *collect_sim*
          5. apply per-sample LoRA delta to the output

        Because the LoRA delta is added to *output*, the correction propagates
        to deeper layers through residual connections and layernorms.
        """
        U_all = self._moe.expert_lora[layer_name]['U']   # [E, d_out, r]
        V_all = self._moe.expert_lora[layer_name]['V']   # [E, r, d_in]
        scaling = self._moe.moe_config.moe_scaling
        proto = self._moe.prototype_matrix[layer_name]   # [E, d_in]
        router = self

        # Hoist per-call/per-layer constants out of the hot path (bit-identical;
        # the mask is fixed for the whole routing pass and proto is fixed per layer).
        mask_bool = attention_mask_ref[0].bool()
        mask_unsq = mask_bool.unsqueeze(-1)                # [batch, seq, 1]
        counts = mask_bool.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [batch, 1]
        proto_f32 = proto.float()
        p_norm = proto.norm(dim=1)                         # [E]

        def hook(module, args, output):
            x = args[0]                               # [batch, seq_len, d_in]

            # --- 1. per-sample mean ---
            x_det = x.detach()
            x_sum = (x_det * mask_unsq).sum(dim=1)     # [batch, d_in]
            x_mean = x_sum / counts                    # [batch, d_in]

            # --- 2. cosine similarity -> expert per sample ---
            x_norm = x_mean.norm(dim=1, keepdim=True)
            dots = x_mean.float() @ proto_f32.T
            sim = dots / (x_norm.float() * p_norm.unsqueeze(0) + 1e-8)
            experts = sim.argmax(dim=1) // router._k  # [batch] task index

            # --- 3. store routing ---
            router._per_sample_routing[layer_name] = experts

            # --- 4. (optional) store similarity for aggregation ---
            if collect_sim is not None:
                collect_sim[layer_name] = sim.float()

            # --- 5. apply per-sample LoRA delta (float32 to match merge_expert_weights) ---
            x_f32 = x.float()
            step1 = torch.matmul(x_f32, V_all[experts].transpose(-1, -2))  # float32
            step2 = torch.matmul(step1, U_all[experts].transpose(-1, -2))  # float32
            delta = (scaling * step2).to(x.dtype)     # single bfloat16 truncation

            return output + delta

        # Navigate to the linear module and register the hook
        parts = layer_name.split('.')
        layer_idx = int(parts[2])
        short_name = '.'.join(parts[3:-1])  # e.g. 'self_attn.q_proj'
        layer_module = self._moe.base_model.model.layers[layer_idx]
        module = layer_module
        for attr in short_name.split('.'):
            module = getattr(module, attr)

        h = module.register_forward_hook(hook)
        hook_list.append(h)
        return h

    def _register_capture_hook(
        self,
        layer_name: str,
        attention_mask_ref: list,
        hook_list: List[torch.utils.hooks.RemovableHandle],
        collect_sim: Dict[str, torch.Tensor],
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook that ONLY captures activations (no LoRA).

        Used during the refinement pass when LoRA hooks are already active via
        ``_register_lora_hooks``.  The capture hook reads the LoRA-corrected
        input, computes cosine similarity vs prototypes, and stores the result
        in *collect_sim* for later aggregation.
        """
        proto = self._moe.prototype_matrix[layer_name]   # [E, d_in]

        # Hoist per-call/per-layer constants (bit-identical).
        mask_bool = attention_mask_ref[0].bool()
        mask_unsq = mask_bool.unsqueeze(-1)
        counts = mask_bool.sum(dim=1).clamp(min=1).unsqueeze(-1)
        proto_f32 = proto.float()
        p_norm = proto.norm(dim=1)

        def hook(module, args, output):
            x = args[0]                               # [batch, seq_len, d_in]

            x_det = x.detach()
            x_sum = (x_det * mask_unsq).sum(dim=1)
            x_mean = x_sum / counts

            x_norm = x_mean.norm(dim=1, keepdim=True)
            dots = x_mean.float() @ proto_f32.T
            sim = dots / (x_norm.float() * p_norm.unsqueeze(0) + 1e-8)
            collect_sim[layer_name] = sim.float()

        parts = layer_name.split('.')
        layer_idx = int(parts[2])
        short_name = '.'.join(parts[3:-1])
        layer_module = self._moe.base_model.model.layers[layer_idx]
        module = layer_module
        for attr in short_name.split('.'):
            module = getattr(module, attr)

        h = module.register_forward_hook(hook)
        hook_list.append(h)
        return h

    # ------------------------------------------------------------------
    # Per-sample LoRA forward hooks (generation phase — read-only)
    # ------------------------------------------------------------------

    def _make_lora_hook(self, layer_name: str):
        """Return a forward hook that adds per-sample LoRA delta to output."""
        U_all = self._moe.expert_lora[layer_name]['U']  # [num_experts, d_out, rank]
        V_all = self._moe.expert_lora[layer_name]['V']  # [num_experts, rank, d_in]
        scaling = self._moe.moe_config.moe_scaling
        router = self  # capture so hook reads live _per_sample_routing dict

        def hook(module, args, output):
            x = args[0]                                  # [batch, seq_len, d_in]
            routing = router._per_sample_routing          # read live dict
            if layer_name not in routing:
                return output
            experts = routing[layer_name]                 # [batch]
            batch = x.shape[0]

            # Handle num_return_sequences expansion during generation:
            # the batch may have been replicated n times from a single prompt.
            if batch != experts.shape[0]:
                repeat = batch // experts.shape[0]
                experts = experts.repeat_interleave(repeat)

            # Compute LoRA delta in float32, then truncate once (matching
            # merge_expert_weights precision).  V/U are stored in float32.
            x_f32 = x.float()
            step1 = torch.matmul(x_f32, V_all[experts].transpose(-1, -2))  # float32
            step2 = torch.matmul(step1, U_all[experts].transpose(-1, -2))  # float32
            delta = (scaling * step2).to(x.dtype)     # single bfloat16 truncation

            return output + delta

        return hook

    def _register_lora_hooks(self):
        """Register per-sample LoRA forward hooks on all routed linear layers."""
        if self._hooks:
            return  # already registered

        moe_start = self._moe.moe_start_layer
        for layer_idx in range(moe_start, self._num_layers):
            for sn in self._routed_short_names:
                layer_name = f'model.layers.{layer_idx}.{sn}.weight'
                if layer_name not in self._moe.expert_lora:
                    continue
                layer_module = self._moe.base_model.model.layers[layer_idx]
                module = layer_module
                for attr in sn.split('.'):
                    module = getattr(module, attr)
                h = module.register_forward_hook(self._make_lora_hook(layer_name))
                self._hooks.append(h)

    def _remove_lora_hooks(self):
        """Remove all registered LoRA hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Dominant expert detection
    # ------------------------------------------------------------------

    def get_dominant_expert(self) -> Optional[str]:
        """
        Return the most frequent expert across all routing decisions.
        Returns None if no routing has been computed.
        """
        if not self._per_sample_routing:
            return None
        from moe_config import TASK_NAMES
        from collections import Counter
        counter = Counter()
        for experts in self._per_sample_routing.values():
            for e in experts.tolist():
                counter[int(e)] += 1
        if not counter:
            return None
        dominant_idx = counter.most_common(1)[0][0]
        return TASK_NAMES[dominant_idx]

    def get_dominant_expert_per_sample(self) -> List[Optional[str]]:
        """Return the modal expert (task name) per sample across all routed layers.

        Used by batched route-then-fix inference: samples sharing a dominant
        expert can be merged once and generated together.
        """
        if not self._per_sample_routing:
            return []
        from collections import Counter
        from moe_config import TASK_NAMES
        batch = next(iter(self._per_sample_routing.values())).shape[0]
        counters = [Counter() for _ in range(batch)]
        for experts in self._per_sample_routing.values():
            for i, e in enumerate(experts.tolist()):
                counters[i][int(e)] += 1
        out: List[Optional[str]] = []
        for c in counters:
            out.append(TASK_NAMES[c.most_common(1)[0][0]] if c else None)
        return out

    def set_expected_expert(self, task_name: str):
        """Set the expected expert for routing accuracy tracking."""
        if task_name in self._moe.task_to_idx:
            self._expected_expert_idx = self._moe.task_to_idx[task_name]
            self._routing_match_count = 0
            self._routing_total_count = 0
            print(f"Routing accuracy tracking enabled: expected expert = '{task_name}'")
        else:
            self._expected_expert_idx = None

    def get_routing_accuracy(self) -> Optional[float]:
        """Return routing accuracy (fraction of decisions routed to expected expert)."""
        if (not hasattr(self, '_expected_expert_idx') or self._expected_expert_idx is None
                or not hasattr(self, '_routing_total_count') or self._routing_total_count == 0):
            return None
        return self._routing_match_count / self._routing_total_count

    def _track_routing_accuracy(self):
        """Accumulate routing accuracy stats from current per-sample routing results."""
        if not hasattr(self, '_expected_expert_idx') or self._expected_expert_idx is None:
            return
        for layer_name, experts in self._per_sample_routing.items():
            for e in experts.tolist():
                self._routing_total_count += 1
                if int(e) == self._expected_expert_idx:
                    self._routing_match_count += 1

    def print_routing_accuracy(self):
        """Print routing accuracy summary."""
        acc = self.get_routing_accuracy()
        if acc is not None:
            expected_task = self._moe.idx_to_task.get(self._expected_expert_idx, 'unknown')
            print(f"\nPer-Sample Routing Accuracy: {self._routing_match_count}/{self._routing_total_count} "
                  f"= {acc*100:.1f}% decisions routed to expected expert '{expected_task}'")
        else:
            print("\nRouting Accuracy: not tracked (no expected expert set)")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_routing_cache(self):
        self._routing_cache.clear()

    def cleanup(self):
        self._remove_lora_hooks()
        self._per_sample_routing.clear()
        self._routing_cache.clear()
