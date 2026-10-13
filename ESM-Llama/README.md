# ESM-Llama: Essential Subspace Merging for Llama-3.2-3B

**ESM (Essential Subspace Merging)** is a training-free method that merges several
task-specific LLMs into a single multi-task model, using only a few unlabeled proxy
samples. **ESM++** extends it with per-sample dynamic routing: a lightweight mixture
of low-rank experts on top of the ESM base.

The five MergeBench domain experts are: `instruction`, `math`, `coding`, `safety`,
`multilingual` (base model `meta-llama/Llama-3.2-3B`).

---

## Requirements

```bash
pip install -r requirements.txt
```

The ESM++ evaluation drivers additionally use three external harnesses, which must be
installed separately and placed on `PYTHONPATH` (or vendored next to this repo):

- [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) — `lm_eval` (gsm8k, ifeval, multilingual)
- [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) — `bigcode_eval` (humanevalplus, mbppplus)
- `safety-eval-fork` — the safety evaluation fork (wildguardtest, harmbench, xstest, …)

---

## Configuration

All paths are read from environment variables (nothing is hardcoded):

| Variable | Used for | Default |
|:--|:--|:--|
| `BASE_MODEL_PATH` | base Llama model (local path or HF id) | `meta-llama/Llama-3.2-3B` |
| `MERGED_MODEL_PATH` | the ESM-fused base consumed by ESM++ | `./merged_models/ESM_scaling_coef_0.9` |
| `PROJECT_ROOT` | this project's root (for aggregation scripts) | the script's directory |
| `HF_HUB_TOKEN` | hub auth, if the datasets/models are gated | — |

```bash
export BASE_MODEL_PATH=/path/to/Llama-3.2-3B
```

Data: the calibration/proxy samples come from the public HuggingFace
[MergeBench](https://huggingface.co/MergeBench) validation sets
(`MergeBench/{instruction,math,coding,safety,multilingual}_val`) and the expert
checkpoints from `MergeBench/Llama-3.2-3B_{task}`. No local dataset layout is needed.

---

## ESM — static merge

1. **Capture per-layer principal directions** (one prompt-only forward per task):

   ```bash
   python capture_layer_stats.py          # writes ./layer_stats.pt
   ```

2. **Merge** (default global scaling coefficient `0.9`; the eigenvalue-weighting
   exponent defaults to `0.15`):

   ```bash
   python ./merging/main.py \
       --algo ESM \
       --base-model "${BASE_MODEL_PATH:-meta-llama/Llama-3.2-3B}" \
       --scaling-coef 0.9 \
       --layer-stats-path ./layer_stats.pt
   ```

   The merged model is written to
   `./merged_models/Llama-3.2-3B_merged/ESM_scaling_coef_0.9/`.

> `merging/main.py` must be run from the project root (`python ./merging/main.py …`);
> it imports its sibling `merging_methods` package relative to the script directory.

---

## ESM++ — dynamic per-sample routing

ESM++ adds *one low-rank expert per task per routed linear layer* (7 projections ×
28 layers) and routes each sample by cosine similarity between the layer-input mean
activation and per-task prototypes. It uses the ESM-fused model as the base.

1. **Capture the experts' principal directions + routing prototypes** from the fused
   base (prompt-only):

   ```bash
   python capture_moe_layer_stats_ponly.py \
       --base-model-path "$MERGED_MODEL_PATH" \
       --layer-stats-path ./moe_layer_stats_ponly.pt \
       --prototype-path  ./moe_prototypes_ponly.pt \
       --num-samples 64
   ```

2. **(Optional) k-means multi-prototype** for better routing (k=8 by default):

   ```bash
   python capture_moe_prototypes_km.py \
       --base-model-path "$MERGED_MODEL_PATH" \
       --k 8 --num-samples 256 \
       --output-path ./moe_prototypes_km8.pt
   ```

3. **Evaluate** (routing is `fix_agg` — see below). One task, via the per-task script:

   ```bash
   MODE=fix_agg TASK=math GPU_ID=0 bash run_moe_routefix_task.sh
   ```

   or all five tasks:

   ```bash
   MODE=fix_agg sbatch run_moe_routefix.sh
   python collect_moe_routefix.py results/moe_routefix    # -> comparison.csv
   ```

   Individual drivers: `eval_moe_lmeval_per_sample.py` (gsm8k/ifeval/multilingual),
   `eval_moe_bigcode_per_sample.py` (coding), `eval_moe_safety_per_sample.py` (safety).

### Defaults

| Arg | Default |
|:--|:--|
| `--prototype-path` | `./moe_prototypes_km8.pt` |
| `--layer-stats-path` | `./moe_layer_stats.pt` |
| `--moe-rank` | `64` |
| `--moe-scaling` | `1.0` |
| `--base-mode` / `--oracle-task` | evaluation baselines (no-expert / forced correct expert) |

---

## Inference: `fix_agg` (the only routing path)

ESM++ routes **once per batch**: it runs an aggregated routing pass on the whole
batch, groups samples by their dominant expert, folds each expert's `U@V` (and its
nonlinear params) into the model weights, and generates each group with a plain
forward — the mathematical equivalent of a per-token LoRA hook, but at base-model
speed (no per-token routing). This is the single routing implementation shipped here;
the slower per-token-hook and per-layer variants have been removed.

---

## Project structure

```
ESM-Llama/
├── merging/
│   ├── main.py                       # ESM merge entry point
│   ├── prepare_args.py
│   └── merging_methods/
│       ├── ESM.py                    # ESM merge algorithm
│       ├── merger.py
│       └── utils.py
├── capture_layer_stats.py            # ESM calibration (principal directions)
├── moe_config.py                     # task registry + MoE config
├── moe_model_for_eval.py             # MoE model: experts, merge/restore, routing apply
├── moe_per_sample_router.py          # per-sample prototype router
├── capture_moe_layer_stats_ponly.py  # ESM++ expert directions + prototypes
├── capture_moe_prototypes_km.py      # ESM++ k-means multi-prototypes
├── eval_moe_lmeval_per_sample.py     # ESM++ eval (gsm8k / ifeval / multilingual)
├── eval_moe_bigcode_per_sample.py    # ESM++ eval (coding)
├── eval_moe_safety_per_sample.py     # ESM++ eval (safety)
├── dispatch_moe_routefix.py / collect_moe_routefix.py
├── run_moe_routefix.sh / run_moe_routefix_task.sh
└── scripts/compare_with_experts.py
```

---

## License

MIT License — see [`../ESM-ViT/LICENSE`](../ESM-ViT/LICENSE).
