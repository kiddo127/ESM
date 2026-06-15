
# ESM-R (Essential Subspace Routing)
source ./scripts.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm

# oracle mode (upper bound)
# CUDA_VISIBLE_DEVICES=7 mode=oracle rank=8 seed=1 merged_model_path=${BASE_DIR}/outs/esm_merged esm_r

CUDA_VISIBLE_DEVICES=7 rank=8 seed=1 prototype_proxy_num=32 merged_model_path=${BASE_DIR}/outs/esm_merged esm_r
