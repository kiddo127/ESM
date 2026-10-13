
# ESM++ (Essential Subspace Routing)
source ./scripts.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm

CUDA_VISIBLE_DEVICES=7 rank=8 seed=1 prototype_proxy_num=32 merged_model_path=outs/esm_merged esm_pp
