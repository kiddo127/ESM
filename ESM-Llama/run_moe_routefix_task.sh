#!/bin/bash
# Run ONE MergeBench task for ONE MoE routing mode (for GPU-pool dispatch).
#
#   MODE = base | oracle | fix_agg
#   TASK = math | instruction | multilingual | coding | safety
#
# Usage:
#   MODE=fix_agg TASK=math GPU_ID=0 bash run_moe_routefix_task.sh

cd "$(dirname "$0")"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export PYTHONUNBUFFERED=1

BASE_MODEL_PATH=${MERGED_MODEL_PATH:-./merged_models/ESM_scaling_coef_0.9}
LAYER_STATS_PATH=${LAYER_STATS_PATH:-./moe_layer_stats_ponly.pt}
PROTOTYPE_PATH=${PROTOTYPE_PATH:-./moe_prototypes_km8.pt}
MOE_RANK=${MOE_RANK:-64}
MOE_SCALING=${MOE_SCALING:-1.0}
MODE=${MODE:?set MODE}
TASK=${TASK:?set TASK}
GPU_ID=${GPU_ID:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results/moe_routefix/$MODE}

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$OUTPUT_ROOT"

MULTI_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

COMMON="--base-model-path $BASE_MODEL_PATH --layer-stats-path $LAYER_STATS_PATH --prototype-path $PROTOTYPE_PATH --moe-rank $MOE_RANK --moe-scaling $MOE_SCALING --gpu-id 0"
case $MODE in
  base)         MODE_ARGS="--base-mode" ;;
  oracle)       MODE_ARGS="" ;;
  fix_agg)      MODE_ARGS="" ;;
  *) echo "Unknown MODE=$MODE"; exit 1 ;;
esac
NEEDS_EXPECTED=1
[ "$MODE" = base ] && NEEDS_EXPECTED=0
[ "$MODE" = oracle ] && NEEDS_EXPECTED=0
EXPERT_FOR_TASK() { case $1 in math) echo math;; instruction) echo instruction;; multilingual) echo multilingual;; coding) echo coding;; safety) echo safety;; esac; }

echo "[routefix] MODE=$MODE TASK=$TASK GPU=$GPU_ID"

case $TASK in
  math|instruction|multilingual)
    conda activate lmeval
    if [ "$TASK" = math ]; then TASKS_ARG="gsm8k_cot"; BS=16
    elif [ "$TASK" = instruction ]; then TASKS_ARG="ifeval"; BS=8
    else TASKS_ARG="$MULTI_TASKS"; BS=8; fi
    python eval_moe_lmeval_per_sample.py $COMMON --tasks $TASKS_ARG --batch-size $BS \
        --output-path $OUTPUT_ROOT/$TASK $MODE_ARGS \
        $([ "$MODE" = oracle ] && echo "--oracle-task $(EXPERT_FOR_TASK $TASK)") \
        $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert $(EXPERT_FOR_TASK $TASK)")
    ;;
  coding)
    conda deactivate; conda activate bigcode
    python eval_moe_bigcode_per_sample.py $COMMON --tasks humanevalplus,mbppplus \
        --n-samples 10 --temperature 0.2 --output-path $OUTPUT_ROOT/code_eval.json $MODE_ARGS \
        $([ "$MODE" = oracle ] && echo "--oracle-task coding") \
        $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert coding")
    ;;
  safety)
    conda deactivate; conda activate safety-eval
    export OPENAI_API_KEY=''
    python eval_moe_safety_per_sample.py $COMMON --tasks wildguardtest,harmbench,xstest,do_anything_now \
        --batch-size 8 --output-path $OUTPUT_ROOT/safety $MODE_ARGS \
        $([ "$MODE" = oracle ] && echo "--oracle-task safety") \
        $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert safety")
    ;;
  *) echo "Unknown TASK=$TASK"; exit 1 ;;
esac
echo "[routefix] done MODE=$MODE TASK=$TASK"
