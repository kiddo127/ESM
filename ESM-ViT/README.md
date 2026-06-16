# ESM-ViT: Essential Subspace Merging for Vision Transformers

**ESM (Essential Subspace Merging)** is a training-free method that merges multiple finetuned ViT models into a single multi-task model via PCA decomposition of output activation displacements.

**ESM++** extends ESM with lightweight LoRA residual experts and per-sample dynamic routing for higher accuracy.

---

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.0, CUDA-capable GPU
- Install dependencies:

```bash
conda env create -f environment.yml
conda activate esm-vit
```

---

## Data & Model Preparation

> **Important** — Before merging, we recommend evaluating each finetuned model on its own task to verify correctness. Compare the results against the reference accuracies in `results/single_task/`. If a single-task result deviates significantly, the checkpoint or data may be misconfigured, and downstream merging results will be unreliable.

Datasets and pretrained/finetuned model checkpoints follow the same structure as the [TSV-M](https://github.com/AntoAndGar/task_singular_vectors) codebase. Please refer to that repository for download instructions.

### Datasets

Place datasets under your `data_location`. The 20 supported tasks are:

| 8-task | +6 (14-task) | +6 (20-task) |
|:---|:---|:---|
| Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SVHN, SUN397 | STL10, OxfordIIITPet, Flowers102, CIFAR100, PCAM, FER2013 | CIFAR10, Food101, FashionMNIST, RenderedSST2, EMNIST, KMNIST |

> **Note on EMNIST** — If the finetuned model performs poorly on EMNIST, it is likely due to a discrepancy in rotation angle between training and test preprocessing. EMNIST images are typically rotated 90° clockwise and flipped horizontally; if your preprocessing does not match, accuracy will degrade. Please verify and fix the data preprocessing in `src/datasets/emnist.py` **before** running model merging.

### Model Checkpoints

For each task, a finetuned checkpoint is expected at:
```
{model_location}/{model}/{Task}Val/nonlinear_finetuned.pt
```
Example: `~/model-merging-models/checkpoints/ViT-B-32/CarsVal/nonlinear_finetuned.pt`

### Configuration

Edit `config/config.yaml` or pass via command line:

```yaml
data_location: "~/data"                              # dataset root
model_location: "~/model-merging-models/checkpoints" # checkpoint root
```

---

## Quick Start

### ESM — Static Fusion

Produces a single merged model with no inference overhead.

```bash
python esm.py model=ViT-B-32 method=ESM num_tasks=8 proxy_n_samples=32 seed=0

# Save the fused model for ESM++ downstream use:
python esm.py model=ViT-B-32 save_main_model=True main_model_path=./fused_8tasks.pt
```

### ESM++ — Dynamic Per-Sample Routing

First run ESM to produce a fused base model, then:

```bash
python esmpp.py model=ViT-B-32 num_tasks=8 lora_rank=32 \
    main_model_path=./fused_8tasks.pt
```

If no fused model is provided, ESM++ falls back to the pretrained checkpoint.

### Key Parameters

| Parameter | Description | Default |
|:---|:---|:---|
| `num_tasks` | Number of tasks (8, 14, 20) | 8 |
| `proxy_n_samples` | Proxy samples per task for ESD | 32 |
| `lora_rank` | LoRA expert rank (ESM++ only) | 32 |
| `seed` | Random seed for proxy sampling | 0 |
| `save_main_model` | Save fused ESM model to disk | False |

### Batch Runs

```bash
bash run_esm.sh      # ESM proxy sample ablation
bash run_esmpp.sh    # ESM++ sweep over models and task counts
```

---

## Project Structure

```
ESM-ViT/
├── esm.py                              # ESM static fusion entry point
├── esmpp.py                            # ESM++ dynamic routing entry point
├── essential_subspace_decomposition.py # ESD core module
├── run_esm.sh / run_esmpp.sh           # Example batch run scripts
├── config/
│   ├── config.yaml                     # Main configuration
│   └── method/ESM.yaml                 # Method-specific parameters
└── src/
    ├── datasets/          (27 files)   # 20 dataset loaders + registry
    ├── eval/
    │   ├── eval.py                     # Evaluation logic
    │   ├── eval_utils.py               # ESM evaluation pipeline
    │   └── aggregation.py              # Task vector creation & merging
    ├── models/
    │   ├── modeling.py                 # ImageEncoder / ImageClassifier
    │   ├── task_vectors.py             # NonLinearTaskVector
    │   ├── heads.py                    # Classification heads
    │   └── esmpp_layers.py             # ESM++ per-sample routing layers
    └── utils/
        ├── esm_utils.py                # ESM merge algorithm
        ├── utils.py                    # General utilities
        ├── variables_and_paths.py      # Path helpers & dataset lists
        └── logging.py                  # WandB logging
```

---

## Acknowledgements

Our code framework is based on [TSV-M (Task Singular Vectors)](https://github.com/AntoAndGar/task_singular_vectors).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
