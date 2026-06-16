#!/bin/bash
# ESM++ (Essential Subspace Merging ++) — Dynamic per-sample routing with LoRA experts
# Each sample independently routed to its best-matching expert via cosine similarity

GPU=0
cd "$(dirname "$0")" || exit 1
mkdir -p results/esmpp

MODELS=("ViT-B-32" "ViT-B-16" "ViT-L-14")
TASK_COUNTS=(8 14 20)

# Model-specific batch sizes (smaller for larger models to avoid OOM)
declare -A BATCH_SIZES
BATCH_SIZES["ViT-B-32"]=128
BATCH_SIZES["ViT-B-16"]=64
BATCH_SIZES["ViT-L-14"]=32

declare -A PCA_BATCH_SIZES
PCA_BATCH_SIZES["ViT-B-32"]=32
PCA_BATCH_SIZES["ViT-B-16"]=32
PCA_BATCH_SIZES["ViT-L-14"]=32

echo "=========================================="
echo "ESM++ (Essential Subspace Merging ++)"
echo "Dynamic Per-Sample Routing with LoRA Experts"
echo "Models: ${MODELS[*]}"
echo "Task counts: ${TASK_COUNTS[*]}"
echo "=========================================="
echo ""

overall_start=$(date +%s)

for MODEL in "${MODELS[@]}"; do
    for NUM_TASKS in "${TASK_COUNTS[@]}"; do
        echo "=========================================="
        echo "Running: model=$MODEL, num_tasks=$NUM_TASKS"
        echo "=========================================="

        BATCH=${BATCH_SIZES[$MODEL]}
        PCA_BATCH=${PCA_BATCH_SIZES[$MODEL]}
        # Set the path to your ESM-fused model (or leave empty to use pretrained)
        # MODEL_PATH="path/to/checkpoints/${MODEL}/fused_model_${NUM_TASKS}tasks.pt"
        MODEL_PATH=""

        if [ -f "$MODEL_PATH" ]; then
            MAIN_MODEL_ARG="main_model_path=$MODEL_PATH"
            echo "  Using fused model: $MODEL_PATH"
        else
            MAIN_MODEL_ARG=""
            echo "  No fused model found, using pretrained"
        fi

        # ----- Per-sample routed mode -----
        ROUTED_OUT="results/esmpp/per_sample_${MODEL}_${NUM_TASKS}tasks.json"
        if [ -f "$ROUTED_OUT" ]; then
            echo "  [per-sample] $ROUTED_OUT already exists, skipping"
        else
            echo "  [per-sample] -> $ROUTED_OUT"
            CUDA_VISIBLE_DEVICES=$GPU python esmpp.py \
                model="$MODEL" num_tasks="$NUM_TASKS" \
                batch_size=$BATCH pca_batch_size=$PCA_BATCH \
                lora_rank=32 oracle_routing=False \
                proxy_n_samples=32 \
                results_save_path="$ROUTED_OUT" \
                $MAIN_MODEL_ARG
        fi

        echo ""
    done
done

overall_end=$(date +%s)
total_min=$(( (overall_end - overall_start) / 60 ))
echo "=========================================="
echo "All done! Total time: ${total_min} minutes"
echo "Results in: results/esmpp/"
ls -lh results/esmpp/
echo "=========================================="
