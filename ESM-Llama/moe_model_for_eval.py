"""
MoE Model for evaluation with ONE-TIME routing per batch.

Key design:
1. Routing happens ONCE per batch, based on INPUT tokens only
2. After routing, expert LoRA parameters are merged into model weights
3. During generation, weights stay fixed (no re-routing)
4. After generation, weights can be restored for next batch
"""

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Optional, Any, List

from moe_config import (
    MODEL_LOCAL,
    TASK_CONFIGS,
    TASK_NAMES,
    MoEConfig,
)


class MoEModelForCausalLM:
    """
    MoE Model with ONE-TIME routing per batch.

    Routing logic:
    1. Before generation: compute routing from input tokens
    2. Merge selected expert LoRA into model weights
    3. Generate with fixed weights (no re-routing)
    4. Restore base weights after generation

    Oracle mode:
    - Directly specify task_id to use corresponding expert for all layers
    - Useful for verification and debugging
    """

    def __init__(self, moe_config: MoEConfig):
        self.moe_config = moe_config
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load base model
        base_model_path = moe_config.get_base_model_path()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map=self._device,
        )
        self.base_model.eval()

        # Store original base weights for restoration (routed linear layers only)
        self.routed_short_names = moe_config.get_routed_layers()
        self.original_weights: Dict[str, torch.Tensor] = {}
        num_layers = self.base_model.config.num_hidden_layers
        for layer_idx in range(num_layers):
            for short_name in self.routed_short_names:
                layer_name = f'model.layers.{layer_idx}.{short_name}.weight'
                self.original_weights[layer_name] = self.base_model.state_dict()[layer_name].clone()

        # Load expert parameters
        self.prototype_matrix: Dict[str, torch.Tensor] = {}
        self.prototype_k: int = 1  # prototypes per task (k>1 => k-means centroids)
        self.expert_lora: Dict[str, Dict[str, torch.Tensor]] = {}
        self.expert_configs: Dict[str, dict] = {}  # task_name -> serialized config
        self.expert_nonlinear: Dict[str, Dict[str, torch.Tensor]] = {}  # task_name -> param_name -> weight (on CPU)
        self._current_nonlinear_expert: Optional[str] = None
        self._load_expert_parameters()

        # Move expert parameters to GPU for fast per-batch access
        self._move_params_to_device()

        # Precompute ordered layer name list (only routed layers)
        self.all_layer_names: List[str] = [
            f'model.layers.{layer_idx}.{short_name}.weight'
            for layer_idx in range(num_layers)
            for short_name in self.routed_short_names
        ]

        # Routing cache for loglikelihood evaluation (hash -> bytes of 196 expert indices)
        self.routing_cache: Dict[int, bytes] = {}

        # Routing results (computed once per batch)
        self.routing_results: Dict[str, int] = {}  # layer_name -> expert_idx
        self.merged_layers: List[str] = []

        # Routing mode
        self.routing_mode = 'auto'  # 'auto' or 'oracle'
        self.oracle_task_id: Optional[str] = None  # task name for oracle mode

        # Routing strategy: 'per_layer' (each layer routes independently)
        #                 or 'per_block' (all layers in a block share one routing decision)
        self.routing_strategy = moe_config.routing_mode

        # First layer index to apply MoE (previous layers use base model only)
        self.moe_start_layer = moe_config.moe_start_layer

        # Copy properties from base model
        self.config = self.base_model.config
        self.generation_config = self.base_model.generation_config

        # Index mappings
        self.task_to_idx = {task: i for i, task in enumerate(TASK_NAMES)}
        self.idx_to_task = {i: task for i, task in enumerate(TASK_NAMES)}

        # Routing accuracy tracking
        self._expected_expert_idx: Optional[int] = None
        self._routing_match_count: int = 0
        self._routing_total_count: int = 0

    def get_module_by_layer_name(self, layer_name: str):
        """Get the nn.Module corresponding to a layer name."""
        parts = layer_name.split('.')
        layer_idx = int(parts[2])
        short_name_parts = parts[3:-1]
        short_name = '.'.join(short_name_parts)

        layer_module = self.base_model.model.layers[layer_idx]
        module = layer_module
        for attr in short_name.split('.'):
            module = getattr(module, attr)
        return module

    def merge_expert_into_model(self, task_id: str):
        """
        Merge a specific expert's LoRA parameters into the base model permanently.

        This is used for oracle mode: merge once at the start, then all
        subsequent forward/generate calls use the merged weights directly.
        No per-batch routing or weight restoration is needed.

        Args:
            task_id: Task name (one of TASK_NAMES)
        """
        expert_idx = self.task_to_idx[task_id]
        print(f"Merging expert '{task_id}' (idx={expert_idx}) into base model permanently...")

        # Merge linear layer LoRA
        merged_count = 0
        for layer_name, orig_weight in self.original_weights.items():
            layer_idx = int(layer_name.split('.')[2])
            if layer_idx < self.moe_start_layer:
                continue
            orig_weight_gpu = orig_weight.to(self._device)
            lora_U = self.expert_lora[layer_name]['U'][expert_idx].to(self._device)
            lora_V = self.expert_lora[layer_name]['V'][expert_idx].to(self._device)
            lora_delta = lora_U @ lora_V
            new_weight = orig_weight_gpu.float() + self.moe_config.moe_scaling * lora_delta
            new_weight = new_weight.to(orig_weight_gpu.dtype)

            module = self.get_module_by_layer_name(layer_name)
            module.weight.data = new_weight
            merged_count += 1

        print(f"  Merged {merged_count} linear layers with expert '{task_id}'")

        # Update model config to match expert config
        if task_id in self.expert_configs:
            expert_cfg = self.expert_configs[task_id]
            # NEVER copy use_cache=False — kills generation speed
            if expert_cfg.get('use_cache', True) == False:
                print(f"  Note: expert '{task_id}' has use_cache=False, keeping use_cache=True for speed")
            # Update generation config (except use_cache)
            gen_cfg = expert_cfg.get('generation_config', {})
            if gen_cfg and hasattr(self.base_model, 'generation_config'):
                for key, value in gen_cfg.items():
                    if key != 'use_cache' and hasattr(self.base_model.generation_config, key):
                        setattr(self.base_model.generation_config, key, value)
            print(f"  Updated generation_config (use_cache unchanged)")

    def apply_expert_config(self, task_id: str):
        """Apply only an expert's generation_config (except use_cache — must stay True for fast generation).
        Lightweight — no weight copying.  Call once before eval starts.
        """
        if task_id not in self.expert_configs:
            print(f"  No config stored for expert '{task_id}'")
            return
        expert_cfg = self.expert_configs[task_id]
        if expert_cfg.get('use_cache', True) == False:
            print(f"  Note: expert '{task_id}' has use_cache=False, keeping use_cache=True for speed")
        gen_cfg = expert_cfg.get('generation_config', {})
        if gen_cfg and hasattr(self.base_model, 'generation_config'):
            for key, value in gen_cfg.items():
                if key != 'use_cache' and hasattr(self.base_model.generation_config, key):
                    setattr(self.base_model.generation_config, key, value)
        print(f"Applied expert '{task_id}' generation_config (use_cache unchanged)")

    def apply_expert_nonlinear(self, task_id: str):
        """Apply embed_tokens, lm_head, and layernorms from a specific expert.
        Idempotent: if the same expert is already applied, does nothing.
        Call ONCE when the dominant expert is determined by routing.
        """
        if task_id not in self.expert_nonlinear:
            return
        if self._current_nonlinear_expert == task_id:
            return  # Already applied
        task_params = self.expert_nonlinear[task_id]
        for param_name, value in task_params.items():
            self._write_model_param(param_name, value.to(self._device))
        self._current_nonlinear_expert = task_id
        # Also apply config
        self.apply_expert_config(task_id)

    def _write_model_param(self, param_name: str, value: torch.Tensor):
        """Write a tensor to a model parameter by name."""
        if 'embed_tokens' in param_name:
            self.base_model.model.embed_tokens.weight.data = value
        elif 'lm_head' in param_name:
            self.base_model.lm_head.weight.data = value
        elif 'model.norm' in param_name:
            self.base_model.model.norm.weight.data = value
        elif 'input_layernorm' in param_name or 'post_attention_layernorm' in param_name:
            parts = param_name.split('.')
            layer_idx = int(parts[2])
            norm_name = parts[3]
            getattr(self.base_model.model.layers[layer_idx], norm_name).weight.data = value

    def get_dominant_expert(self) -> Optional[str]:
        """
        Return the most frequent expert across all routing decisions.
        Returns None if no routing has been computed.
        """
        if not self.routing_results:
            return None
        from collections import Counter
        counter = Counter()
        for expert_idx in self.routing_results.values():
            counter[expert_idx] += 1
        if not counter:
            return None
        dominant_idx = counter.most_common(1)[0][0]
        return self.idx_to_task.get(dominant_idx, None)

    def set_expected_expert(self, task_name: str):
        """Set the expected expert for routing accuracy tracking."""
        if task_name in self.task_to_idx:
            self._expected_expert_idx = self.task_to_idx[task_name]
            self._routing_match_count = 0
            self._routing_total_count = 0
            print(f"Routing accuracy tracking enabled: expected expert = '{task_name}'")
        else:
            self._expected_expert_idx = None

    def get_routing_accuracy(self) -> Optional[float]:
        """Return routing accuracy (fraction of layers routed to expected expert)."""
        if self._expected_expert_idx is None or self._routing_total_count == 0:
            return None
        return self._routing_match_count / self._routing_total_count

    def _track_routing_accuracy(self):
        """Accumulate routing accuracy stats from current routing results."""
        if self._expected_expert_idx is None:
            return
        for layer_name, expert_idx in self.routing_results.items():
            self._routing_total_count += 1
            if expert_idx == self._expected_expert_idx:
                self._routing_match_count += 1

    def print_routing_accuracy(self):
        """Print routing accuracy summary."""
        acc = self.get_routing_accuracy()
        if acc is not None:
            expected_task = self.idx_to_task.get(self._expected_expert_idx, 'unknown')
            print(f"\nRouting Accuracy: {self._routing_match_count}/{self._routing_total_count} "
                  f"= {acc*100:.1f}% layers routed to expected expert '{expected_task}'")
        else:
            print("\nRouting Accuracy: not tracked (no expected expert set)")

    def set_oracle_mode(self, task_id: str):
        """
        Set oracle routing mode.

        Args:
            task_id: Task name (one of TASK_NAMES: instruction, math, coding, safety, multilingual)
        """
        if task_id not in TASK_NAMES:
            raise ValueError(f"Invalid task_id: {task_id}. Must be one of {TASK_NAMES}")
        self.routing_mode = 'oracle'
        self.oracle_task_id = task_id
        print(f"Oracle mode enabled: using expert '{task_id}' for all layers")

    def set_auto_mode(self):
        """Set automatic routing mode (compute routing from input)."""
        self.routing_mode = 'auto'
        self.oracle_task_id = None
        print("Auto routing mode enabled")

    def set_base_mode(self):
        """Set base mode - use only base model without any expert LoRA."""
        self.routing_mode = 'base'
        self.oracle_task_id = None
        print("Base mode enabled: using base model only (no expert LoRA)")

    @property
    def device(self):
        return self._device

    def _load_expert_parameters(self):
        """Load expert LoRA parameters and non-linear parameter deltas."""
        layer_stats = torch.load(self.moe_config.layer_stats_path, map_location='cpu', weights_only=True)
        prototypes_raw = torch.load(self.moe_config.prototype_path, map_location='cpu', weights_only=True)

        # Load base state dict (CPU, bfloat16 — kept for non-linear delta computation)
        base_sd_full = {k: v.cpu() for k, v in self.base_model.state_dict().items()}
        num_layers = self.base_model.config.num_hidden_layers

        # Load fine-tuned models (on CPU) and their configs
        ft_sds = {}
        for task_name in TASK_NAMES:
            ft_model_path = TASK_CONFIGS[task_name]['ft_model']
            try:
                ft_model = AutoModelForCausalLM.from_pretrained(ft_model_path, dtype=torch.bfloat16, device_map='cpu')
            except TypeError:
                ft_model = AutoModelForCausalLM.from_pretrained(ft_model_path, device_map='cpu')
            ft_sds[task_name] = {k: v.cpu() for k, v in ft_model.state_dict().items()}
            self.expert_configs[task_name] = {
                'use_cache': ft_model.config.use_cache,
                'generation_config': ft_model.generation_config.to_dict() if ft_model.generation_config else {},
                'vocab_size': ft_model.config.vocab_size,
            }
            del ft_model
        gc.collect()

        rank = self.moe_config.moe_rank

        for layer_idx in range(num_layers):
            for short_name in self.routed_short_names:
                layer_name = f'model.layers.{layer_idx}.{short_name}.weight'

                base_weight = base_sd_full[layer_name].float()  # CPU
                d_out, d_in = base_weight.shape

                # Prototypes: single mean [d_in] -> [num_experts, d_in];
                # OR k-means centroids [k, d_in] -> [num_experts * k, d_in]
                # (task-major: prototype index p maps to task p // k).
                proto_list = [prototypes_raw[task][layer_name] for task in TASK_NAMES]
                if proto_list[0].dim() == 1:
                    self.prototype_k = 1
                    self.prototype_matrix[layer_name] = torch.stack(proto_list, dim=0)
                else:
                    self.prototype_k = int(proto_list[0].shape[0])
                    self.prototype_matrix[layer_name] = torch.cat(proto_list, dim=0)

                # Stack U and V
                U_list = []
                V_list = []
                for task_name in TASK_NAMES:
                    eigenvectors = layer_stats[task_name][layer_name]['offset_eigenvectors'].float()
                    ft_weight = ft_sds[task_name][layer_name].float()
                    delta_W = ft_weight - base_weight

                    k = min(rank, eigenvectors.shape[1])
                    U = eigenvectors[:, :k]
                    V = eigenvectors[:, :k].T @ delta_W

                    U_list.append(U)
                    V_list.append(V)

                self.expert_lora[layer_name] = {
                    'U': torch.stack(U_list, dim=0),  # [num_experts, d_out, rank]
                    'V': torch.stack(V_list, dim=0),  # [num_experts, rank, d_in]
                }

        # Free the large stats file (86 GB) — eigenvectors no longer needed
        del layer_stats, prototypes_raw
        gc.collect()

        # Store non-linear params (embed, lm_head, norms) for each expert
        # These are applied ONCE when the dominant expert is determined by routing
        nonlinear_names = ['model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight']
        for layer_idx in range(num_layers):
            nonlinear_names.append(f'model.layers.{layer_idx}.input_layernorm.weight')
            nonlinear_names.append(f'model.layers.{layer_idx}.post_attention_layernorm.weight')

        for task_name in TASK_NAMES:
            task_params = {}
            for param_name in nonlinear_names:
                if param_name in ft_sds[task_name]:
                    ft_val = ft_sds[task_name][param_name]
                    # Handle vocab size mismatch for embed/lm_head
                    if 'embed_tokens' in param_name or 'lm_head' in param_name:
                        shared = min(ft_val.shape[0], base_sd_full[param_name].shape[0])
                        ft_val = ft_val[:shared]
                    task_params[param_name] = ft_val.cpu().clone()
            self.expert_nonlinear[task_name] = task_params

        for task_name in TASK_NAMES:
            del ft_sds[task_name]
        del base_sd_full
        gc.collect()

        print(f'Expert parameters loaded for {len(self.expert_lora)} layers '
              f'+ {len(nonlinear_names)} non-linear params per expert')

    def _move_params_to_device(self):
        """
        Move expert parameters to GPU for fast per-batch access.

        original_weights is already on GPU (cloned from GPU-resident state_dict),
        but expert_lora and prototype_matrix are loaded on CPU.
        """
        for layer_name in self.expert_lora:
            self.expert_lora[layer_name]['U'] = self.expert_lora[layer_name]['U'].to(self._device)
            self.expert_lora[layer_name]['V'] = self.expert_lora[layer_name]['V'].to(self._device)
        for layer_name in self.prototype_matrix:
            self.prototype_matrix[layer_name] = self.prototype_matrix[layer_name].to(self._device)

    @staticmethod
    def _compute_seq_cache_key(input_ids: torch.Tensor, attention_mask: torch.Tensor, seq_idx: int) -> int:
        """
        Compute a cache key for a single sequence in a batch.

        Uses the first 80% of non-padding tokens so that the same question
        with different continuations (e.g. MMLU's A/B/C/D options) maps to
        the same cache key.
        """
        valid_mask = attention_mask[seq_idx].bool()
        valid_tokens = input_ids[seq_idx][valid_mask]
        prefix_len = max(1, int(valid_tokens.shape[0] * 0.8))
        prefix = valid_tokens[:prefix_len]
        return hash(prefix.cpu().numpy().tobytes())

    def _get_routing_layer_for_block(self, layer_idx: int, block_input_dim: int) -> Optional[str]:
        """Find first routed layer in block with compatible prototype dimension."""
        for short_name in self.routed_short_names:
            layer_name = f'model.layers.{layer_idx}.{short_name}.weight'
            if layer_name in self.prototype_matrix:
                proto_dim = self.prototype_matrix[layer_name].shape[-1]
                if proto_dim == block_input_dim:
                    return layer_name
        return None

    def _compute_routing_from_mean(self, x_mean: torch.Tensor, layer_name: str) -> int:
        """Compute expert (task) index from mean activation vector and layer prototypes.

        With k prototypes per task the prototype matrix is [num_experts * k, d_in]
        (task-major), so the winning prototype index is divided by k to get the task.
        """
        proto_matrix = self.prototype_matrix[layer_name]  # already on GPU
        x_norm = x_mean.norm()
        proto_norms = proto_matrix.norm(dim=1)
        if x_norm > 1e-8:
            dots = x_mean @ proto_matrix.T
            similarities = dots / (x_norm * proto_norms + 1e-8)
        else:
            similarities = torch.zeros(proto_matrix.shape[0], device=self._device)
        return similarities.argmax().item() // self.prototype_k

    def compute_routing_from_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        """
        Compute routing decisions for all layers based on INPUT tokens only.

        In auto mode: uses forward pass to capture each layer's input activations.
          If use_cache=True and attention_mask is provided, the routing result is
          cached by the first-80%-prefix of each sequence's input IDs, so that
          identical questions with different continuations reuse the same routing.
        In oracle mode: directly uses the specified task_id for all layers.
        In base mode: skip routing, use base model only.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            use_cache: Whether to use/update the routing cache
        """
        # Base mode: skip routing entirely
        if self.routing_mode == 'base':
            self.routing_results = {}
            print("Base mode: skipping routing, using base model only")
            return

        # Oracle mode: directly set routing results
        if self.routing_mode == 'oracle' and self.oracle_task_id is not None:
            expert_idx = self.task_to_idx[self.oracle_task_id]
            num_layers = self.base_model.config.num_hidden_layers
            self.routing_results = {}
            for layer_idx in range(self.moe_start_layer, num_layers):
                for short_name in self.routed_short_names:
                    layer_name = f'model.layers.{layer_idx}.{short_name}.weight'
                    self.routing_results[layer_name] = expert_idx
            print(f"Oracle routing: layers {self.moe_start_layer}-{num_layers-1} -> {self.oracle_task_id}")
            return

        # Auto mode: check cache first
        cache_keys = None
        if use_cache and attention_mask is not None:
            cache_keys = tuple(
                self._compute_seq_cache_key(input_ids, attention_mask, i)
                for i in range(input_ids.shape[0])
            )
            if all(k in self.routing_cache for k in cache_keys):
                cached = self.routing_cache[cache_keys[0]]
                self.routing_results = {
                    name: cached[i] for i, name in enumerate(self.all_layer_names)
                }
                print("Routing: cached hit")
                return

        # Auto mode: compute routing from input activations
        num_layers = self.base_model.config.num_hidden_layers
        current_attention_mask = [attention_mask]

        if self.routing_strategy == 'per_block':
            # --- Per-block routing: capture normalized block inputs, route once per block ---
            block_inputs: Dict[int, torch.Tensor] = {}

            def make_layernorm_capture_hook(layer_idx: int):
                def hook(module, args, output):
                    x = output.detach()  # post-layernorm [batch, seq, d_model]
                    mask = current_attention_mask[0]
                    if mask is not None:
                        mask_bool = mask.bool()
                        x_flat = x.reshape(-1, x.shape[-1])
                        mask_flat = mask_bool.reshape(-1)
                        x_valid = x_flat[mask_flat]
                        x_mean = x_valid.mean(dim=0)
                    else:
                        x_mean = x.mean(dim=(0, 1))
                    block_inputs[layer_idx] = x_mean.float()
                return hook

            handles = []
            for layer_idx in range(self.moe_start_layer, num_layers):
                layernorm = self.base_model.model.layers[layer_idx].input_layernorm
                handle = layernorm.register_forward_hook(make_layernorm_capture_hook(layer_idx))
                handles.append(handle)

            with torch.no_grad():
                _ = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            for h in handles:
                h.remove()

            self.routing_results = {}
            expert_counts = {task: 0 for task in TASK_NAMES}

            for layer_idx, x_mean in block_inputs.items():
                routing_layer = self._get_routing_layer_for_block(layer_idx, x_mean.shape[-1])
                if routing_layer is None:
                    best_idx = 0
                else:
                    best_idx = self._compute_routing_from_mean(x_mean, routing_layer)
                for short_name in self.routed_short_names:
                    layer_name = f'model.layers.{layer_idx}.{short_name}.weight'
                    self.routing_results[layer_name] = best_idx
                expert_counts[self.idx_to_task[best_idx]] += 1

        else:
            # --- Per-layer routing: capture each linear layer's input, route independently ---
            layer_inputs: Dict[str, torch.Tensor] = {}

            def make_input_capture_hook(layer_idx: int, short_name: str):
                layer_name = f'model.layers.{layer_idx}.{short_name}.weight'

                def hook(module, args, output):
                    x = args[0].detach()  # Input to this layer [batch, seq, d_in]
                    mask = current_attention_mask[0]

                    if mask is not None:
                        mask_bool = mask.bool()
                        x_flat = x.reshape(-1, x.shape[-1])
                        mask_flat = mask_bool.reshape(-1)
                        x_valid = x_flat[mask_flat]
                        x_mean = x_valid.mean(dim=0)
                    else:
                        x_mean = x.mean(dim=(0, 1))

                    layer_inputs[layer_name] = x_mean.float()

                return hook

            handles = []
            for layer_idx in range(self.moe_start_layer, num_layers):
                for short_name in self.routed_short_names:
                    layer_module = self.base_model.model.layers[layer_idx]
                    module = layer_module
                    for attr in short_name.split('.'):
                        module = getattr(module, attr)
                    handle = module.register_forward_hook(make_input_capture_hook(layer_idx, short_name))
                    handles.append(handle)

            with torch.no_grad():
                _ = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            for h in handles:
                h.remove()

            self.routing_results = {}
            expert_counts = {task: 0 for task in TASK_NAMES}

            for layer_name, x_mean in layer_inputs.items():
                best_idx = self._compute_routing_from_mean(x_mean, layer_name)
                self.routing_results[layer_name] = best_idx
                expert_counts[self.idx_to_task[best_idx]] += 1

        # Print routing summary
        print(f"Routing: {dict(zip(TASK_NAMES, [expert_counts[t] for t in TASK_NAMES]))}")

        # Track routing accuracy
        self._track_routing_accuracy()

        # Cache results for future reuse
        if use_cache and cache_keys is not None:
            routing_bytes = bytes(self.routing_results[name] for name in self.all_layer_names)
            for k in cache_keys:
                if k not in self.routing_cache:
                    self.routing_cache[k] = routing_bytes

    def merge_expert_weights(self):
        """
        Merge selected expert LoRA parameters into model weights.

        After this call, model weights are modified and will stay modified
        until restore_base_weights() is called.

        In base mode: skip merging entirely, weights stay as original.
        """
        # Base mode: no merging needed
        if self.routing_mode == 'base':
            print("Base mode: skipping weight merge, using original base weights")
            return

        if not self.routing_results:
            raise RuntimeError("No routing results. Call compute_routing_from_input() first.")

        for layer_name, expert_idx in self.routing_results.items():
            orig_weight = self.original_weights[layer_name]  # already on GPU
            lora_U = self.expert_lora[layer_name]['U'][expert_idx]  # already on GPU
            lora_V = self.expert_lora[layer_name]['V'][expert_idx]  # already on GPU
            lora_delta = lora_U @ lora_V
            new_weight = orig_weight.float() + self.moe_config.moe_scaling * lora_delta
            new_weight = new_weight.to(orig_weight.dtype)

            module = self.get_module_by_layer_name(layer_name)
            module.weight.data = new_weight

            self.merged_layers.append(layer_name)

        print(f"Merged {len(self.merged_layers)} layers with expert LoRA")

    def restore_base_weights(self):
        """
        Restore original base model weights.

        Call this after generation to prepare for next batch.
        """
        if not self.merged_layers:
            return

        for layer_name in self.merged_layers:
            orig_weight = self.original_weights[layer_name]  # already on GPU
            module = self.get_module_by_layer_name(layer_name)
            module.weight.data = orig_weight

        self.merged_layers = []
        self.routing_results = {}
        print("Restored base weights")

    def apply_per_sample_routing(self, per_sample_routing: Dict[str, torch.Tensor]):
        """Set self.routing_results from a PerSampleRouter's per-sample routing.

        Enables single-sample route-then-fix inference (batch=1): the selected
        expert for sample 0 of every routed layer becomes the routing decision,
        which merge_expert_weights() then folds into the model weights.

        Args:
            per_sample_routing: {layer_name: LongTensor[batch]} from PerSampleRouter.
        """
        self.routing_results = {
            layer_name: int(idx.reshape(-1)[0].item())
            for layer_name, idx in per_sample_routing.items()
        }
        return self.routing_results

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass.

        In auto mode: weights should already be merged before calling forward.
        In oracle mode: weights are permanently merged, just forward directly.
        """
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """
        Generate.

        In auto mode: 1. route  2. merge  3. generate  4. restore
        In oracle mode: weights are permanently merged, just generate.
        """
        if self.routing_mode in ('oracle', 'base'):
            return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Auto mode: route, merge, generate, restore
        self.compute_routing_from_input(input_ids, attention_mask)
        self.merge_expert_weights()
        outputs = self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        self.restore_base_weights()
        return outputs

    def cleanup(self):
        """Clean up resources."""
        self.restore_base_weights()
        self.prototype_matrix = {}
        self.expert_lora = {}
        self.original_weights = {}
        gc.collect()
        torch.cuda.empty_cache()


def create_moe_model_for_eval(moe_config: MoEConfig):
    """Create MoE model for evaluation."""
    model = MoEModelForCausalLM(moe_config)
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create MoE model for evaluation')
    parser.add_argument('--base-model-path', type=str, default='')
    parser.add_argument('--layer-stats-path', type=str, default='./moe_layer_stats.pt')
    parser.add_argument('--prototype-path', type=str, default='./moe_prototypes.pt')
    parser.add_argument('--moe-rank', type=int, default=64)
    parser.add_argument('--moe-scaling', type=float, default=1.0)
    parser.add_argument('--routing-mode', type=str, default='per_layer',
                        choices=['per_layer', 'per_block'],
                        help="Routing mode: 'per_layer' (default) or 'per_block'")
    args = parser.parse_args()

    moe_config = MoEConfig(
        base_model_path=args.base_model_path,
        moe_rank=args.moe_rank,
        moe_scaling=args.moe_scaling,
        layer_stats_path=args.layer_stats_path,
        prototype_path=args.prototype_path,
        routing_mode=args.routing_mode,
    )

    model = create_moe_model_for_eval(moe_config)
    tokenizer = AutoTokenizer.from_pretrained(moe_config.get_base_model_path())
    tokenizer.pad_token = tokenizer.eos_token

    print('MoE model created successfully!')
    print(f'Model config: {model.config}')

    # Test generation
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(f'Test output: {tokenizer.decode(outputs[0])}')

    model.cleanup()