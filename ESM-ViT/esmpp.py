#!/usr/bin/env python3
"""
ESM++ (Essential Subspace Merging ++) — Dynamic per-sample routing with LoRA experts.

Builds on ESM's essential subspace merging by adding per-sample dynamic routing:
each sample is independently routed to its best-matching LoRA expert using cosine
similarity to task prototypes.

Key differences from esm.py (static fusion):
  - Uses MoERoutedLinearPerSample / MoEMultiheadAttentionPerSample
  - Imports from src.models.esmpp_layers
  - Records per-layer routing accuracy during evaluation
"""

import json
import os
import time
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.task_vectors import NonLinearTaskVector
from src.models.modeling import ImageEncoder, ImageClassifier
from src.models.heads import get_classification_head
from src.models.esmpp_layers import (
    LoRAExpert,
    build_moe_model_per_sample,
    set_forced_expert_per_sample,
    set_recording_per_sample,
    get_all_routing_decisions_per_sample,
)
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.utils import utils
from src.utils.variables_and_paths import (
    ALL_DATASETS,
    get_finetuned_path,
    get_single_task_accuracies_path,
    get_zeroshot_path,
)


# ---------------------------------------------------------------------------
# Principal direction computation (SVD on activation differences)
# ---------------------------------------------------------------------------
def _svd_principal_direction(act_data):
    original_shape = act_data.shape
    feature_dim = original_shape[-1]
    if act_data.dim() > 2:
        act_data = act_data.reshape(-1, feature_dim)
    n_samples = act_data.shape[0]
    try:
        U, S, Vh = torch.linalg.svd(act_data, full_matrices=False)
        eigenvalues = S ** 2 / (n_samples - 1)
    except:
        U, S, Vh = torch.linalg.svd(act_data * 0.1, full_matrices=False)
        eigenvalues = S ** 2 / (n_samples - 1) * 100
    eigenvectors = Vh.T
    if eigenvectors.shape[1] < feature_dim:
        full_eig = torch.eye(feature_dim, device=act_data.device, dtype=act_data.dtype)
        full_eig[:, :eigenvectors.shape[1]] = eigenvectors
        eigenvectors = full_eig
        full_eigenvalues = torch.zeros(feature_dim, device=eigenvalues.device, dtype=eigenvalues.dtype)
        full_eigenvalues[:len(eigenvalues)] = eigenvalues
        eigenvalues = full_eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    return eigenvalues, eigenvectors


def _make_pd_hook(layer_key, delta_W, pd_dict):
    def hook(module, input_tensor, output_tensor):
        x = input_tensor[0]
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        activation_diff = torch.matmul(x, delta_W.T.to(x.device)).detach()
        eigenvalues, eigenvectors = _svd_principal_direction(activation_diff)
        pd_dict[layer_key] = {
            'eigenvalues': eigenvalues.cpu(),
            'eigenvectors': eigenvectors.cpu(),
        }
    return hook


def _make_attn_pd_hook(module_name, delta_W_qkv, pd_dict):
    def hook(module, input_tensor, output_tensor):
        x = input_tensor[0]
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        qkv_diff = torch.matmul(x, delta_W_qkv.T.to(x.device)).detach()
        eigenvalues, eigenvectors = _svd_principal_direction(qkv_diff)
        pd_dict[module_name + '.in_proj_weight'] = {
            'eigenvalues': eigenvalues.cpu(),
            'eigenvectors': eigenvectors.cpu(),
        }
    return hook


# ---------------------------------------------------------------------------
# Prototype collector
# ---------------------------------------------------------------------------
class _PrototypeCollector:
    def __init__(self):
        self.sum_acts = None
        self.count = 0

    def __call__(self, module, input_tensor, output_tensor):
        x = input_tensor[0]
        if x.dim() == 3:
            x_flat = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            x_flat = x
        else:
            return
        if self.sum_acts is None:
            self.sum_acts = x_flat.sum(dim=0).cpu()
        else:
            self.sum_acts += x_flat.sum(dim=0).cpu()
        self.count += x_flat.shape[0]

    def get_prototype(self):
        if self.count > 0 and self.sum_acts is not None:
            return self.sum_acts / self.count
        return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _param_key_to_module_name(key: str) -> str:
    if key.endswith('.in_proj_weight'):
        return key[:-len('.in_proj_weight')]
    if key.endswith('.weight'):
        return key[:-len('.weight')]
    return key


# ---------------------------------------------------------------------------
# Single-pass: principal directions + prototypes per task
# ---------------------------------------------------------------------------
def _forward_and_collect(cfg, ft_state_dicts, task_vector_dict,
                         routed_param_keys, device):
    pd_dict_all = {}
    proto_dict_all = {key: {} for key in routed_param_keys}

    for dataset in cfg.DATASETS_VAL:
        task_name = dataset.replace("Val", "")
        print(f"\n{'='*70}")
        print(f"[{task_name}] Forwarding finetuned model (1 batch)")

        dt_name = dataset.replace("Val", "")
        ft_path = get_finetuned_path(cfg.model_location, dt_name, cfg.model)
        finetuned_encoder = ImageEncoder.load(cfg.model, ft_path)
        finetuned_encoder = finetuned_encoder.to(device)

        task_vec = task_vector_dict[dataset].vector

        pd_dict_all[dataset] = {}
        proto_collectors = []

        for key in routed_param_keys:
            mod_name = _param_key_to_module_name(key)
            try:
                module = finetuned_encoder.get_submodule(mod_name)
            except AttributeError:
                continue

            delta_W = task_vec.get(key)
            if delta_W is not None:
                if key.endswith('.in_proj_weight'):
                    hook = _make_attn_pd_hook(mod_name, delta_W, pd_dict_all[dataset])
                else:
                    hook = _make_pd_hook(key, delta_W, pd_dict_all[dataset])
                module.register_forward_hook(hook)

            col = _PrototypeCollector()
            module.register_forward_hook(col)
            proto_collectors.append((key, col))

        classification_head = get_classification_head(cfg, dataset)
        classifier = ImageClassifier(finetuned_encoder, classification_head)
        classifier.eval()

        dataset_obj = get_dataset(
            dataset,
            classifier.val_preprocess,
            location=cfg.data_location,
            batch_size=cfg.pca_batch_size,
        )

        proxy_target = cfg.get("proxy_n_samples", 128)
        seed = cfg.get("seed", 0)

        # Use seeded random subset of exactly proxy_target samples
        test_ds = dataset_obj.test_dataset
        n_available = len(test_ds)
        n_samples = min(proxy_target, n_available)
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_available, size=n_samples, replace=False)

        from torch.utils.data import Subset, DataLoader
        subset = Subset(test_ds, indices.tolist())
        dataloader = DataLoader(subset, batch_size=cfg.pca_batch_size, shuffle=False)

        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = maybe_dictionarize(batch_data)
                x = batch_data["images"].to(device)
                _ = utils.get_logits(x, classifier)

        for key, col in proto_collectors:
            proto = col.get_prototype()
            if proto is not None:
                proto_dict_all[key][task_name] = proto

        n_pd = len(pd_dict_all[dataset])
        print(f"  Collected {n_pd} principal direction entries, "
              f"{sum(1 for _, c in proto_collectors if c.get_prototype() is not None)} prototypes")

    return pd_dict_all, proto_dict_all


# ---------------------------------------------------------------------------
# Extract LoRA experts from principal directions
# ---------------------------------------------------------------------------
def _extract_lora_experts(pd_dict_all, task_vector_dict, routed_param_keys, rank, device):
    routed_experts = {key: {} for key in routed_param_keys}

    for dataset, task_vector in task_vector_dict.items():
        task_name = dataset.replace("Val", "")
        pd_dict = pd_dict_all.get(dataset, {})

        for key in routed_param_keys:
            if key not in pd_dict:
                continue
            if key not in task_vector.vector:
                continue

            delta_W = task_vector.vector[key].to(device)
            eigenvectors = pd_dict[key]['eigenvectors'].to(device)

            actual_rank = min(rank, eigenvectors.shape[1], delta_W.shape[0], delta_W.shape[1])
            B = eigenvectors[:, :actual_rank]
            A = B.T @ delta_W

            routed_experts[key][task_name] = LoRAExpert(A.cpu(), B.cpu())

    return routed_experts


# ===================================================================
# Evaluation with routing accuracy
# ===================================================================
def eval_single_dataset_with_routing(image_encoder, dataset_name, args):
    """Evaluate on a single dataset while recording per-layer routing decisions.

    Returns (metrics, routing_per_layer) where:
      - metrics: {"top1": float}
      - routing_per_layer: {layer_name: {"indices": tensor, "task_names": list}}
    """
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    set_recording_per_sample(image_encoder, True)

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n

    set_recording_per_sample(image_encoder, False)

    routing_decisions = get_all_routing_decisions_per_sample(image_encoder)

    metrics = {"top1": top1}
    print(
        f"Done evaluating on {dataset_name}.\t Accuracy: {100*top1:.2f}%."
    )
    return metrics, routing_decisions


# ===================================================================
# Main
# ===================================================================
@hydra.main(config_path="config", config_name="config", version_base="1.3")
def esmpp(cfg: DictConfig) -> None:
    # ---- Setup ----------------------------------------------------------
    if cfg.DATASETS == "":
        cfg.DATASETS = ALL_DATASETS[:cfg.num_tasks]
    else:
        cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [d + "Val" for d in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "save_dir", os.path.join(cfg.model_location, cfg.model), force_add=True)

    device = torch.device(cfg.device)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rank = cfg.get("lora_rank", 64)
    print(f"device={device}, lora_rank={rank}, seed={seed}, num_tasks={cfg.num_tasks}")
    print(f"Datasets: {cfg.DATASETS}")
    print(f"Routing mode: per-sample")

    # ---- Phase 0: Load main model ---------------------------------------
    print("\n" + "=" * 70)
    print("Phase 0: Loading main model")
    pretrained_checkpoint = get_zeroshot_path(cfg.model_location, "MNIST", cfg.model)
    main_model_path = cfg.get("main_model_path", "")

    if main_model_path:
        print(f"  Using saved fused model: {main_model_path}")
        main_encoder = ImageEncoder.load(cfg.model, main_model_path)
    else:
        print("  Using original pretrained model")
        pt_ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
        main_encoder = ImageEncoder(cfg.model)
        if hasattr(pt_ckpt, 'state_dict'):
            pt_ckpt = pt_ckpt.state_dict()
        main_encoder.load_state_dict(pt_ckpt)
    main_encoder = main_encoder.to(device)

    main_state_dict_cpu = {k: v.cpu() for k, v in main_encoder.state_dict().items()}

    # ---- Determine routed layer keys -----------------------------------
    routed_param_keys = sorted([
        k for k in main_state_dict_cpu.keys()
        if k.endswith('attn.in_proj_weight') or k.endswith('mlp.c_fc.weight')
    ])
    print(f"\nRouted layers ({len(routed_param_keys)}):")
    for k in routed_param_keys:
        print(f"  {k}")

    # ---- Phase 1a: Compute task vectors ----------------------------------
    print("\n" + "=" * 70)
    print("Phase 1a: Computing task vectors (relative to main model)")
    task_vector_dict = {}
    ft_state_dicts = {}

    for dataset in cfg.DATASETS_VAL:
        dt_name = dataset.replace("Val", "")
        ft_path = get_finetuned_path(cfg.model_location, dt_name, cfg.model)
        ft_sd = torch.load(ft_path, map_location="cpu")

        delta_W = {}
        for key in main_state_dict_cpu:
            if key in ft_sd and main_state_dict_cpu[key].shape == ft_sd[key].shape:
                if main_state_dict_cpu[key].dtype in (torch.int64, torch.uint8):
                    continue
                delta_W[key] = ft_sd[key] - main_state_dict_cpu[key]

        task_vector_dict[dataset] = NonLinearTaskVector(cfg.model, vector=delta_W)
        ft_state_dicts[dataset] = ft_sd
        n_routed = sum(1 for k in delta_W if 'in_proj_weight' in k or 'c_fc.weight' in k)
        print(f"  {dt_name}: ΔW computed ({n_routed} routed layers)")

    # ---- Phase 1b+2+3: Single forward pass per task --------------------
    print("\n" + "=" * 70)
    print("Phase 1b+2+3: Single forward pass per task")
    print("  (finetuned model forward + PD hooks + prototype collection)")
    pd_dict_all, proto_dict_all = _forward_and_collect(
        cfg, ft_state_dicts, task_vector_dict, routed_param_keys, device,
    )

    # ---- Phase 4: Extract LoRA experts ----------------------------------
    print("\n" + "=" * 70)
    print("Phase 4: Extracting LoRA experts from principal directions")
    routed_experts = _extract_lora_experts(
        pd_dict_all, task_vector_dict, routed_param_keys, rank, device,
    )
    for key in routed_param_keys:
        n_exp = len(routed_experts.get(key, {}))
        print(f"  {key}: {n_exp} experts")

    # ---- Phase 6: Build per-sample MoE model ----------------------------
    print("\n" + "=" * 70)
    print("Phase 6: Assembling per-sample MoE model")
    moe_model = build_moe_model_per_sample(main_encoder, routed_experts, proto_dict_all)
    moe_model.eval()

    # ---- Phase 7: Evaluation -------------------------------------------
    print("\n" + "=" * 70)
    print("*" * 24, "Evaluating Per-Sample MoE Model on Test Sets", "*" * 24)
    eval_start = time.time()

    if "num_workers" not in cfg or cfg.num_workers is None:
        OmegaConf.update(cfg, "num_workers", 2, force_add=True)

    ft_accuracies_path = get_single_task_accuracies_path(cfg.model)
    with open(ft_accuracies_path) as f:
        ft_accuracies = json.load(f)

    use_oracle = cfg.get("oracle_routing", False)
    per_dataset_results = {}
    all_routing_decisions = {}  # dataset_name -> {layer_name: {"indices": tensor, "task_names": list}}

    for dataset_name in cfg.DATASETS:
        if use_oracle:
            set_forced_expert_per_sample(moe_model, dataset_name)
            results = eval_single_dataset(moe_model, dataset_name, cfg)
        else:
            set_forced_expert_per_sample(moe_model, None)
            results, routing_decisions = eval_single_dataset_with_routing(
                moe_model, dataset_name, cfg
            )
            all_routing_decisions[dataset_name] = routing_decisions

        top1 = results["top1"]
        normalized_top1 = top1 / ft_accuracies[dataset_name]
        per_dataset_results[dataset_name + ":top1"] = top1
        per_dataset_results[dataset_name + ":normalized_top1"] = normalized_top1
        routing_mode = "oracle" if use_oracle else "per-sample"
        print(f"  [{routing_mode:10s}] {dataset_name:20s}  top1={100*top1:5.2f}%  normalized={100*normalized_top1:5.2f}%")

    # ---- Compute routing accuracy (route mode only) ----------------------
    routing_accuracy = {}
    if not use_oracle and all_routing_decisions:
        # Get all layer names from the first dataset's routing decisions
        first_dataset = next(iter(all_routing_decisions))
        layer_names = sorted(all_routing_decisions[first_dataset].keys())

        for layer_name in layer_names:
            routing_accuracy[layer_name] = {}
            for dataset_name in cfg.DATASETS:
                rd = all_routing_decisions.get(dataset_name, {})
                layer_info = rd.get(layer_name)
                if layer_info is None:
                    routing_accuracy[layer_name][dataset_name] = None
                    continue
                indices = layer_info["indices"]
                task_names = layer_info["task_names"]
                # Map index -> task name, check if it matches the ground truth dataset
                correct = sum(
                    1 for idx in indices.tolist()
                    if task_names[idx] == dataset_name
                )
                total = len(indices)
                routing_accuracy[layer_name][dataset_name] = round(correct / total, 4) if total > 0 else None

        # Print per-layer routing accuracy summary
        print("\n" + "-" * 70)
        print("Per-layer routing accuracy (%):")
        print(f"  {'Layer':<55s} " + " ".join(f"{d[:6]:>7s}" for d in cfg.DATASETS))
        for layer_name in layer_names:
            short_name = layer_name.replace("model.visual.transformer.resblocks.", "block_")
            vals = [f"{routing_accuracy[layer_name][d]*100:6.1f}%" if routing_accuracy[layer_name][d] is not None else "    N/A" for d in cfg.DATASETS]
            print(f"  {short_name:<55s} " + " ".join(vals))
        print("-" * 70)

    eval_total = time.time() - eval_start
    avg_top1 = float(np.mean([per_dataset_results[d + ":top1"] for d in cfg.DATASETS]))
    avg_normalized_top1 = float(np.mean([per_dataset_results[d + ":normalized_top1"] for d in cfg.DATASETS]))
    print(f"\n  {'Average':20s}  top1={100*avg_top1:5.2f}%  normalized={100*avg_normalized_top1:5.2f}%")
    print(f"  {'Total time':20s}  {eval_total:.2f}s")

    print("\n" + "=" * 70)
    print("Per-sample MoE model evaluation complete!")
    print(f"  Main model: {'fused' if main_model_path else 'pretrained'}")
    print(f"  Routed layers: {len(routed_param_keys)}")
    print(f"  Experts per layer: {cfg.num_tasks}")
    print(f"  LoRA rank: {rank}")

    # ---- Save results --------------------------------------------------
    save_path = cfg.get("results_save_path", "")
    if save_path:
        routing_mode = "oracle" if use_oracle else "per_sample"
        save_data = {
            "model": cfg.model,
            "num_tasks": cfg.num_tasks,
            "lora_rank": rank,
            "main_model": "fused" if main_model_path else "pretrained",
            "routing_mode": routing_mode,
            "total_time_sec": round(eval_total, 2),
            "avg_top1": round(100 * avg_top1, 2),
            "avg_normalized_top1": round(100 * avg_normalized_top1, 2),
        }
        for d in cfg.DATASETS:
            save_data[f"{d}:top1"] = round(100 * per_dataset_results[d + ":top1"], 2)
            save_data[f"{d}:normalized_top1"] = round(100 * per_dataset_results[d + ":normalized_top1"], 2)

        if routing_accuracy:
            save_data["routing_accuracy"] = routing_accuracy

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    esmpp()
