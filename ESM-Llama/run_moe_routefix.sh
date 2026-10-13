#!/bin/bash
#SBATCH --job-name=moe_routefix
#SBATCH --partition=lvjq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
#SBATCH -e %j.err
#
# Run all 5 MergeBench tasks for ONE MoE routing mode.
#
#   MODE = base | oracle | fix_agg
#
# Override with env vars, e.g.:
#   MODE=fix_agg sbatch run_moe_routefix.sh
#
# Output: results/moe_routefix/$MODE/{math,multilingual,ifeval,code,safety}/

cd "$(dirname "$0")"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1

BASE_MODEL_PATH=${MERGED_MODEL_PATH:-./merged_models/ESM_scaling_coef_0.9}
LAYER_STATS_PATH=${LAYER_STATS_PATH:-./moe_layer_stats_ponly.pt}
PROTOTYPE_PATH=${PROTOTYPE_PATH:-./moe_prototypes_km8.pt}
MOE_RANK=${MOE_RANK:-64}
MOE_SCALING=${MOE_SCALING:-1.0}
MODE=${MODE:-fix_agg}
GPU_ID=${GPU_ID:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-./results/moe_routefix/$MODE}

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p $OUTPUT_ROOT

MULTI_TASKS="m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru"

# Mode -> driver flags
case $MODE in
  base)          MODE_ARGS_COMMON="--base-mode" ;;
  oracle)        MODE_ARGS_COMMON="" ;;  # --oracle-task set per task below
  fix_agg)       MODE_ARGS_COMMON="" ;;
  *) echo "Unknown MODE=$MODE"; exit 1 ;;
esac

# Modes that route automatically (track routing accuracy + expected expert)
NEEDS_EXPECTED=1
[ "$MODE" = "base" ] && NEEDS_EXPECTED=0
[ "$MODE" = "oracle" ] && NEEDS_EXPECTED=0

echo "======================================================================"
echo " MoE route-fix eval | MODE=$MODE | base=$BASE_MODEL_PATH"
echo "======================================================================"
START=$(date +%s)

# ============ Task 1+2+3: lm_eval (gsm8k / multilingual / ifeval) ============
conda activate lmeval

python eval_moe_lmeval_per_sample.py \
    --base-model-path $BASE_MODEL_PATH --layer-stats-path $LAYER_STATS_PATH \
    --prototype-path $PROTOTYPE_PATH --moe-rank $MOE_RANK --moe-scaling $MOE_SCALING \
    --tasks gsm8k_cot --batch-size 16 --output-path $OUTPUT_ROOT/math --gpu-id 0 \
    $MODE_ARGS_COMMON \
    $([ "$MODE" = oracle ] && echo "--oracle-task math") \
    $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert math")

python eval_moe_lmeval_per_sample.py \
    --base-model-path $BASE_MODEL_PATH --layer-stats-path $LAYER_STATS_PATH \
    --prototype-path $PROTOTYPE_PATH --moe-rank $MOE_RANK --moe-scaling $MOE_SCALING \
    --tasks $MULTI_TASKS --batch-size 8 --output-path $OUTPUT_ROOT/multilingual --gpu-id 0 \
    $MODE_ARGS_COMMON \
    $([ "$MODE" = oracle ] && echo "--oracle-task multilingual") \
    $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert multilingual")

python eval_moe_lmeval_per_sample.py \
    --base-model-path $BASE_MODEL_PATH --layer-stats-path $LAYER_STATS_PATH \
    --prototype-path $PROTOTYPE_PATH --moe-rank $MOE_RANK --moe-scaling $MOE_SCALING \
    --tasks ifeval --batch-size 8 --output-path $OUTPUT_ROOT/instruction --gpu-id 0 \
    $MODE_ARGS_COMMON \
    $([ "$MODE" = oracle ] && echo "--oracle-task instruction") \
    $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert instruction")

# ============ Task 4: bigcode (humanevalplus, mbppplus) ============
conda deactivate
conda activate bigcode

python eval_moe_bigcode_per_sample.py \
    --base-model-path $BASE_MODEL_PATH --layer-stats-path $LAYER_STATS_PATH \
    --prototype-path $PROTOTYPE_PATH --moe-rank $MOE_RANK --moe-scaling $MOE_SCALING \
    --tasks humanevalplus,mbppplus --n-samples 10 --temperature 0.2 \
    --output-path $OUTPUT_ROOT/code_eval.json --gpu-id 0 \
    $MODE_ARGS_COMMON \
    $([ "$MODE" = oracle ] && echo "--oracle-task coding") \
    $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert coding")

# ============ Task 5: safety ============
conda deactivate
conda activate safety-eval
export OPENAI_API_KEY=''

python eval_moe_safety_per_sample.py \
    --base-model-path $BASE_MODEL_PATH --layer-stats-path $LAYER_STATS_PATH \
    --prototype-path $PROTOTYPE_PATH --moe-rank $MOE_RANK --moe-scaling $MOE_SCALING \
    --tasks wildguardtest,harmbench,xstest,do_anything_now \
    --batch-size 8 --output-path $OUTPUT_ROOT/safety --gpu-id 0 \
    $MODE_ARGS_COMMON \
    $([ "$MODE" = oracle ] && echo "--oracle-task safety") \
    $([ "$NEEDS_EXPECTED" = 1 ] && echo "--expected-expert safety")

echo "======================================================================"
echo " MODE=$MODE done. Total: $(( $(date +%s) - START ))s  -> $OUTPUT_ROOT"
echo "======================================================================"
