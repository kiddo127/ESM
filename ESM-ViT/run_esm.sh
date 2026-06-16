#!/bin/bash
# ESM (Essential Subspace Merging) — Static fusion into a single merged model

GPU=0
cd "$(dirname "$0")" || exit 1

# Example runs (uncomment as needed):
CUDA_VISIBLE_DEVICES=$GPU python esm.py model=ViT-B-32 method="ESM" num_tasks=8 proxy_n_samples=32 seed=0
# CUDA_VISIBLE_DEVICES=$GPU python esm.py model=ViT-B-16 method="ESM" num_tasks=14 proxy_n_samples=32 seed=0
# CUDA_VISIBLE_DEVICES=$GPU python esm.py model=ViT-L-14 method="ESM" num_tasks=20 proxy_n_samples=32 seed=0
