# ESM-LM: Essential Subspace Mixture for Language Models

This repository implements two complementary methods for multi-task model merging:

- **ESM-M** (Essential Subspace Mixture — Merging): Fuses multiple task-specific models into a single merged model via subspace decomposition and Procrustes orthogonalization.
- **ESM-R** (Essential Subspace Mixture — Routing): Builds a dynamic Mixture of Experts from task-specific models, routing each input to the best expert via prototype-based cosine similarity.

Based on [RoBERTa](https://arxiv.org/abs/1907.11692) fine-tuned on 8 GLUE tasks.

## Project Structure

```
discriminative_moe_clean/
├── config/
│   ├── esm_m.yml              # ESM-M merge config
│   ├── esm_r.yml              # ESM-R routing config
│   └── glue.py                # GLUE evaluation metric
├── run_merge.py               # Main entry point
├── merge.py                   # Core merge/routing algorithms
├── esm_moe_eval.py            # ESM-R dynamic router + evaluator
├── essential_subspace_decomposition.py  # PCA / principal direction
├── search_scaling.py          # Optimal scaling coefficient search
├── param.py                   # Parameter arithmetic wrapper
├── eval.py                    # GLUE evaluation utilities
├── utils.py                   # I/O, data loading, helpers
├── sparsify.py                # Sparsification utilities
├── scripts.sh                 # Shell functions (esm_m, esm_r)
├── run_esm.sh                 # ESM-M example run script
├── run_esm_r.sh               # ESM-R example run script
└── requirements.txt           # Python dependencies
```

## Requirements

```bash
conda create -n esm python=3.10
conda activate esm
pip install -r requirements.txt
```

Key packages: PyTorch >= 2.0, Transformers >= 4.30, Datasets >= 2.14.

## Directory Setup

This is a **code-only** repository — all paths are relative to the project root.  Place data, models, and outputs alongside the code:

```
ESM-LM/
├── config/
├── run_merge.py
├── ...
├── data/
│   ├── test.json              # GLUE test data
│   └── validation.json        # Proxy data for principal direction / prototypes
├── outs/
│   ├── merge_results/         # Results output (created automatically)
│   └── esm_merged/            # ESM-M merged model (created after merge)
└── ../roberta/                # Finetuned models (relative to project root)
    └── {task}/
        └── roberta-base_lr1e-05/
            ├── config.json
            ├── model.safetensors
            └── classifier_head.pt
```

For roberta-large, use `../roberta_large/{task}/roberta-large_lr1e-05/`.

### Preparing Finetuned Models

Fine-tune RoBERTa-base on each GLUE task and save in the structure above. Each model directory needs:
- `config.json` / `model.safetensors` — standard HuggingFace format
- `classifier_head.pt` — saved classifier head (`torch.save(model.classifier, ...)`)

### Preparing Data

- `data/test.json` — list of `{"input_ids": [...], "attention_mask": [...], "label": ..., "dataset_ids": ...}`
- `data/validation.json` — same format, used for computing principal directions and collecting prototypes

## Usage

### ESM-M (Merging)

Fuse 8 task-specific models into one merged checkpoint, then evaluate:

```bash
bash run_esm.sh
```

Or with custom parameters:

```bash
source scripts.sh
CUDA_VISIBLE_DEVICES=0 seed=0 prototype_proxy_num=32 esm_m
```

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `select_merge` | 8 | Number of tasks to merge |
| `seed` | 10 | Random seed |
| `prototype_proxy_num` | 32 | Proxy samples per task for PCA |
| `principal_data_path` | `$BASE_DIR/data/validation.json` | Data for principal direction |
| `save_path` | `$BASE_DIR/outs/esm_merged` | Merged model output path |

After merging, the model is saved to `save_path` and evaluated on `data/test.json`.

### ESM-R (Routing)

Use the ESM-M merged model as base, extract per-task experts, and route dynamically:

```bash
bash run_esm_r.sh
```

Or with custom parameters:

```bash
source scripts.sh
CUDA_VISIBLE_DEVICES=0 rank=8 seed=0 prototype_proxy_num=32 \
    merged_model_path=$BASE_DIR/outs/esm_merged esm_r
```

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 8 | Expert low-rank dimension (0 = auto d/T) |
| `mode` | `route` | Routing mode: `route` / `oracle` / `base` |
| `merged_model_path` | `$BASE_DIR/outs/esm_merged` | Base model for MoE |
| `prototype_proxy_num` | 32 | Proxy samples per task for routing prototypes |

**Modes:**
- `route` — prototype-based cosine similarity routing (default)
- `oracle` — directly use the correct expert (upper bound performance)
- `base` — no routing, run the base merged model directly

### Output

Results are saved to `$BASE_DIR/outs/merge_results/results.csv` and `results.md`, including per-task and average scores.

ESM-R additionally saves routing accuracy to `routing_accuracy_rank{rank}_seed{seed}.json`.
