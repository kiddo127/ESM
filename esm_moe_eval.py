import torch
import torch.nn.functional as F
import random
import tqdm
from collections import defaultdict


class ESMMoEEvaluator:
    """
    Evaluator for ESM-DMoE.

    Three modes:
      - mode="route" (default): each layer routes independently via cosine similarity
        with task prototypes. Picks the highest-scoring expert.
      - mode="oracle": uses the pre-set current_oracle_task to directly apply the
        correct expert's ΔW. Skips routing computation entirely.
      - mode="base": no routing, no experts. Runs the base model directly.

    In route/oracle modes, the expert is applied as LoRA-style ΔW (U@V), not the full
    finetuned weights.
    """

    def __init__(self, base_model, expert_dict=None, prototype_dict=None, mode="route",
                 record_routing=True):
        self.base_model = base_model
        self.expert_dict = expert_dict
        self._attention_mask = None
        self.mode = mode
        self.current_oracle_task = None

        self.prototype_dict = prototype_dict
        self.record_routing = record_routing
        self.current_task = None
        self.routing_records = []  # list of {module_name, task, winner, sim}

        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype

        if mode == "base":
            # Base mode: no experts, no routing, no patching
            self._original_forwards = {}
            self._module_map = {}
            return

        # Map: module_name -> experts dict
        self.module_to_experts = {}
        for param_name, experts in expert_dict.items():
            if param_name.endswith('.weight'):
                module_name = param_name[:-7]
                self.module_to_experts[module_name] = experts

        # Map: module_name -> prototype tensors (normalized, shape: [T, d_in])
        self.module_to_prototypes = {}
        if prototype_dict is not None:
            for param_name, task_protos in prototype_dict.items():
                if param_name.endswith('.weight'):
                    module_name = param_name[:-7]
                    self.module_to_prototypes[module_name] = task_protos

        # Pre-build module lookup for fast weight access and forward patching
        self._module_map = dict(base_model.named_modules())
        self._original_forwards = {}

        self._preprocess_experts(device, dtype)
        self._preprocess_prototypes(device, dtype)

        # Monkey-patch nn.Linear.forward on target modules
        self._patch_linears()

    # ------------------------------------------------------------------
    #  Preprocessing
    # ------------------------------------------------------------------
    def _preprocess_experts(self, device, dtype):
        """Move experts to device, compute ΔW=U@V per task per layer."""
        self.module_to_delta_w = {}
        self.module_to_task_names = {}

        for module_name, experts in self.module_to_experts.items():
            task_names = list(experts.keys())
            delta_w_list = []

            for task_name in task_names:
                exp = experts[task_name]
                exp['U'] = exp['U'].to(device=device, dtype=dtype)
                exp['V'] = exp['V'].to(device=device, dtype=dtype)
                delta_w_list.append(exp['U'] @ exp['V'])

            self.module_to_delta_w[module_name] = delta_w_list
            self.module_to_task_names[module_name] = task_names

        # Pre-compute fused weights as stacked tensors [T, d_out, d_in]
        self._fused_stacked = {}    # {module_name: [T, d_out, d_in]}
        self._orig_weights = {}     # {module_name: original weight}
        for module_name, delta_w_list in self.module_to_delta_w.items():
            orig_w = self._module_map[module_name].weight.data
            self._orig_weights[module_name] = orig_w
            fused_list = [orig_w + dw for dw in delta_w_list]
            self._fused_stacked[module_name] = torch.stack(fused_list, dim=0)

        # Oracle: pre-compute per-task fused weights {task_name: {module_name: W_fused}}
        self._oracle_fused = {}
        for module_name, task_names in self.module_to_task_names.items():
            delta_w_list = self.module_to_delta_w[module_name]
            orig_w = self._orig_weights[module_name]
            for tn, dw in zip(task_names, delta_w_list):
                if tn not in self._oracle_fused:
                    self._oracle_fused[tn] = {}
                self._oracle_fused[tn][module_name] = orig_w + dw

    def _preprocess_prototypes(self, device, dtype):
        """Prepare normalized prototype tensors for routing."""
        if self.prototype_dict is None:
            return
        for param_name, task_protos in self.prototype_dict.items():
            if not param_name.endswith('.weight'):
                continue
            module_name = param_name[:-7]
            if module_name not in self.module_to_prototypes:
                continue
            experts = self.module_to_experts.get(module_name, {})
            task_names = list(experts.keys()) if experts else sorted(task_protos.keys())
            proto_list = []
            for tname in task_names:
                proto = task_protos[tname].to(device=device, dtype=dtype)
                # proto shape: [1, d_in] (mean-pooled, n_clusters=1)
                proto_norm = F.normalize(proto.squeeze(0), dim=0)  # [d_in]
                proto_list.append(proto_norm)
            # Stack to [T, d_in]
            self.module_to_prototypes[module_name] = torch.stack(proto_list, dim=0)

    # ------------------------------------------------------------------
    #  Monkey-patch nn.Linear.forward
    # ------------------------------------------------------------------
    def _patch_linears(self):
        for module_name in self.module_to_experts:
            module = self._module_map[module_name]
            self._original_forwards[module_name] = module.forward

            module.forward = self._make_patched_forward(
                module, module.forward, module_name,
            )

    def _make_patched_forward(self, module, orig_forward, module_name):
        fused_stacked = self._fused_stacked[module_name]
        evaluator = self

        if self.mode == "oracle":
            task_names = self.module_to_task_names[module_name]

            # Oracle: directly use the expert for the current task, no routing
            def patched_forward_oracle(input):
                fused = evaluator._oracle_fused[evaluator.current_oracle_task][module_name]
                saved = module.weight.data
                module.weight.data = fused
                output = orig_forward(input)
                module.weight.data = saved

                if evaluator.record_routing and evaluator.current_task is not None:
                    oracle_idx = task_names.index(evaluator.current_oracle_task)
                    evaluator.routing_records.append({
                        'module': module_name,
                        'task': evaluator.current_task,
                        'winner': oracle_idx,
                        'sim': [1.0 if i == oracle_idx else 0.0 for i in range(len(task_names))],
                        'winner_name': evaluator.current_oracle_task,
                    })

                return output

            return patched_forward_oracle

        # Normal mode: route via cosine similarity with prototypes
        proto_tensor = self.module_to_prototypes.get(module_name, None)

        task_names = self.module_to_task_names[module_name]

        def patched_forward(input):
            # Route: mean-pool all non-padding tokens across the batch
            mask = evaluator._attention_mask.unsqueeze(-1).float()
            masked_x = input * mask
            total_tokens = mask.sum().clamp(min=1)
            x_pooled = masked_x.sum(dim=(0, 1)) / total_tokens  # [d_in]
            x_pooled_norm = F.normalize(x_pooled, dim=0)         # [d_in]

            # Cosine similarity with each task prototype: [T]
            sim = x_pooled_norm @ proto_tensor.T

            # Pick highest-scoring expert
            winner = sim.argmax()

            if evaluator.record_routing and evaluator.current_task is not None:
                evaluator.routing_records.append({
                    'module': module_name,
                    'task': evaluator.current_task,
                    'winner': winner.item(),
                    'sim': sim.detach().cpu().tolist(),
                    'winner_name': task_names[winner.item()],
                })

            # Pointer swap weight, forward, restore
            saved = module.weight.data
            module.weight.data = fused_stacked[winner]
            output = orig_forward(input)
            module.weight.data = saved
            return output

        return patched_forward

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, input_ids, attention_mask):
        self._attention_mask = attention_mask
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def clear_hooks(self):
        if hasattr(self, '_original_forwards'):
            for module_name, orig_forward in self._original_forwards.items():
                self._module_map[module_name].forward = orig_forward
            self._original_forwards.clear()

    def reset_routing_records(self):
        self.routing_records.clear()

    def get_routing_accuracy(self):
        """Compute routing accuracy per layer per task.

        Returns:
            dict: {module_name: {task_name: {'accuracy': float, 'correct': int, 'total': int}}}
        """
        stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

        for record in self.routing_records:
            module = record['module']
            task = record['task']
            winner_name = record['winner_name']

            stats[module][task]['total'] += 1
            if winner_name == task:
                stats[module][task]['correct'] += 1

        accuracy = {}
        for module_name in sorted(stats.keys()):
            accuracy[module_name] = {}
            for task_name in sorted(stats[module_name].keys()):
                counts = stats[module_name][task_name]
                acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
                accuracy[module_name][task_name] = {
                    'accuracy': acc,
                    'correct': counts['correct'],
                    'total': counts['total'],
                }

        return accuracy

    def print_routing_accuracy(self):
        """Print routing accuracy summary per layer per task."""
        accuracy = self.get_routing_accuracy()
        if not accuracy:
            print("No routing records available.")
            return

        # Collect all task names
        all_tasks = sorted(set(
            task for module_stats in accuracy.values()
            for task in module_stats.keys()
        ))

        # Header
        header = f"{'Layer':<45}" + "".join(f"{t:<12}" for t in all_tasks) + f"{'Avg':<12}"
        print("\n" + "=" * len(header))
        print("Routing Accuracy (per layer, per task)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Per layer
        layer_avgs = []
        for module_name, task_stats in accuracy.items():
            short_name = module_name.replace('roberta.encoder.layer.', 'L').replace('encoder.layer.', 'L')
            row = f"{short_name:<45}"
            layer_accs = []
            for t in all_tasks:
                if t in task_stats:
                    acc = task_stats[t]['accuracy']
                    layer_accs.append(acc)
                    row += f"{acc*100:>8.1f}%   "
                else:
                    row += f"{'N/A':<12}"
            layer_avg = sum(layer_accs) / len(layer_accs) if layer_accs else 0.0
            layer_avgs.append(layer_avg)
            row += f"{layer_avg*100:>8.1f}%"
            print(row)

        # Footer: per-task average across layers
        print("-" * len(header))
        task_avgs = []
        footer = f"{'Avg':<45}"
        for t in all_tasks:
            task_accs = [accuracy[m][t]['accuracy'] for m in accuracy if t in accuracy[m]]
            task_avg = sum(task_accs) / len(task_accs) if task_accs else 0.0
            task_avgs.append(task_avg)
            footer += f"{task_avg*100:>8.1f}%   "
        overall_avg = sum(task_avgs) / len(task_avgs) if task_avgs else 0.0
        footer += f"{overall_avg*100:>8.1f}%"
        print(footer)
        # Compute micro-average
        total_correct = sum(
            accuracy[m][t]['correct'] for m in accuracy for t in accuracy[m])
        total_samples = sum(
            accuracy[m][t]['total'] for m in accuracy for t in accuracy[m])
        micro_avg = (total_correct / total_samples * 100) if total_samples > 0 else 0.0

        print("=" * len(header))
        print(f"Overall routing accuracy: {overall_avg*100:.2f}% "
              f"({micro_avg:.2f}% micro-average)")
        print()

        return accuracy

    def save_routing_accuracy(self, filepath):
        """Save routing accuracy to a JSON file."""
        accuracy = self.get_routing_accuracy()
        # Convert to serializable format
        serializable = {}
        for module_name, task_stats in accuracy.items():
            serializable[module_name] = {}
            for task_name, counts in task_stats.items():
                serializable[module_name][task_name] = {
                    'accuracy': round(counts['accuracy'], 6),
                    'correct': counts['correct'],
                    'total': counts['total'],
                }
        with open(filepath, 'w') as f:
            import json
            json.dump(serializable, f, indent=2)
        print(f"Routing accuracy saved to {filepath}")


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
def collect_prototypes(
    target_layer_names,
    finetuned_models,
    val_data_path="data/validation.json",
    proxy_num=64,
    device="cuda:0",
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
        model = finetuned_models[data_name]

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

        for weight_name, feature_list in collected.items():
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