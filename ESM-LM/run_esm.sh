
# ESM (Essential Subspace Merging)
source ./scripts.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm

CUDA_VISIBLE_DEVICES=6 seed=1 prototype_proxy_num=32 esm
