"""ESD (Essential Subspace Decomposition) — PCA on output activation displacement.

Core mathematical module for ESM and ESM++. For each task and layer, computes
the principal directions (eigenvectors) of the activation difference ΔO = X ΔW^T,
ordered by descending eigenvalues (λ = σ²/(n-1)).
"""

import torch
import numpy as np

from src.models.task_vectors import NonLinearTaskVector
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path

from src.models.heads import get_classification_head
from src.models.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.utils import utils
from torch.utils.data import DataLoader


def _compute_esd(act_data):
    """SVD-based PCA on activation difference.

    Returns eigenvalues λ = σ²/(n-1) and eigenvectors (principal directions),
    sorted by descending eigenvalue.
    """
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


def _make_esd_hook(layer_name, delta_W, esd_dict=None):
    """Forward hook for MLP layers: computes ESD on activation displacement."""
    def hook(module, input_tensor, output_tensor):
        key = layer_name + '.weight'
        x = input_tensor[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        activation_diff = torch.matmul(x, delta_W.T.to(x.device)).detach()
        eigenvalues, eigenvectors = _compute_esd(activation_diff)
        esd_dict[key] = {
            'eigenvalues': eigenvalues.cpu(),
            'eigenvectors': eigenvectors.cpu()
        }
    return hook


def _make_attn_esd_hook(module_name, W_qkv, b_qkv, delta_W_qkv, delta_W_o, esd_dict=None):
    """Forward hook for attention layers: computes ESD on QKV and O-projection displacement."""
    def hook(module, input_tensor, output_tensor):
        x = input_tensor[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        # QKV projection ESD
        qkv_diff = torch.matmul(x, delta_W_qkv.T.to(x.device)).detach()
        qkv_eigenvalues, qkv_eigenvectors = _compute_esd(qkv_diff)
        esd_dict[module_name + '.in_proj_weight'] = {
            'eigenvalues': qkv_eigenvalues.cpu(),
            'eigenvectors': qkv_eigenvectors.cpu()
        }

        # # O-projection ESD
        # W_q, W_k, W_v = torch.chunk(W_qkv, 3, dim=0)
        # b_q, b_k, b_v = torch.chunk(b_qkv, 3, dim=0)
        # o_input = (torch.matmul(x, W_v.T.to(x.device)) + b_v.to(x.device)).detach()
        # o_diff = torch.matmul(o_input, delta_W_o.T.to(x.device)).detach()
        # o_eigenvalues, o_eigenvectors = _compute_esd(o_diff)
        # esd_dict[module_name + '.out_proj.weight'] = {
        #     'eigenvalues': o_eigenvalues.cpu(),
        #     'eigenvectors': o_eigenvectors.cpu()
        # }
    return hook


def _make_proj_esd_hook(layer_name, delta_W, esd_dict=None):
    """Forward hook for final projection layer."""
    def hook(module, input_tensor, output_tensor):
        key = layer_name.replace('.ln_post', '.proj')
        x = output_tensor
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        activation_diff = (x @ delta_W.to(x.device)).detach()
        eigenvalues, eigenvectors = _compute_esd(activation_diff)
        esd_dict[key] = {
            'eigenvalues': eigenvalues.cpu(),
            'eigenvectors': eigenvectors.cpu()
        }
    return hook


def esd(cfg, task_vector_dict):
    """Essential Subspace Decomposition for all tasks and target layers.

    For each task, loads the finetuned model, registers forward hooks on attention,
    MLP, and projection layers, and computes PCA on the activation displacement
    ΔO = X ΔW^T to obtain principal directions.

    Returns:
        {dataset_name: {layer_key: {eigenvalues, eigenvectors}}}
    """
    esd_dict = {}

    pretrained_checkpoint = get_zeroshot_path(cfg.model_location, "MNIST", cfg.model)

    for val_dataset in cfg.DATASETS_VAL:
        dataset_name = val_dataset.replace("Val", "")
        print(f"\n{'=' * 70}")
        print(f"ESD on {dataset_name}")

        task_vectors = task_vector_dict[val_dataset]
        esd_dict[val_dataset] = {}

        finetuned_checkpoint = get_finetuned_path(cfg.model_location, dataset_name, cfg.model)
        task_vector = NonLinearTaskVector(cfg.model, pretrained_checkpoint, finetuned_checkpoint)

        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0, args=cfg)

        # Collect target layer names
        target_layer_names = []
        for name, _ in image_encoder.named_parameters():
            if "weight" in name and ".mlp" in name:
                target_layer_names.append(name.replace('.weight', ''))
            # elif name.endswith(".proj"):  # skip: single class token insufficient for decomposition
            #     target_layer_names.append(name.replace('.proj', '.ln_post'))
        for name, _ in image_encoder.named_modules():
            if name.endswith(".attn"):
                target_layer_names.append(name)
        print(f"  Target layers: {len(target_layer_names)}")

        # Register forward hooks
        cnt = 0
        finetuned_param_dict = dict(image_encoder.named_parameters())
        for name, module in image_encoder.named_modules():
            if name in target_layer_names:
                cnt += 1
                if name.endswith(".attn"):
                    module.register_forward_hook(
                        _make_attn_esd_hook(
                            module_name=name,
                            W_qkv=finetuned_param_dict[name + '.in_proj_weight'],
                            b_qkv=finetuned_param_dict[name + '.in_proj_bias'],
                            delta_W_qkv=task_vectors.vector[name + '.in_proj_weight'],
                            delta_W_o=task_vectors.vector[name + '.out_proj.weight'],
                            esd_dict=esd_dict[val_dataset]))
                elif name.endswith(".ln_post"):
                    pass  # skip: single class token insufficient for decomposition
                else:
                    weight_name = name + ".weight"
                    module.register_forward_hook(
                        _make_esd_hook(name, task_vectors.vector[weight_name], esd_dict[val_dataset]))
        assert cnt == len(target_layer_names)

        # Forward pass to collect activations
        classification_head = get_classification_head(cfg, val_dataset)
        model = ImageClassifier(image_encoder, classification_head)
        model.eval()
        dataset = get_dataset(
            val_dataset,
            model.val_preprocess,
            location=cfg.data_location,
            batch_size=cfg.pca_batch_size,
        )

        proxy_n_samples = cfg.get("proxy_n_samples", cfg.pca_batch_size)
        seed = cfg.get("seed", 0)
        test_ds = dataset.test_dataset
        n_available = len(test_ds)
        n_samples = min(proxy_n_samples, n_available)
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_available, size=n_samples, replace=False)

        from torch.utils.data import Subset
        subset = Subset(test_ds, indices.tolist())
        dataloader = DataLoader(subset, batch_size=n_samples, shuffle=False)

        device = cfg.device
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                data = maybe_dictionarize(data)
                x = data["images"].to(device)
                logits = utils.get_logits(x, model)
                break

        print(f"  {dataset_name} ESD complete.")

    return esd_dict
