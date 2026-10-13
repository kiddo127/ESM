"""
Capture MULTI-prototype routing centroids (k-means) per task per linear layer.

The single-prototype router uses the mean prompt-only input activation per task
per layer as one prototype.  This script instead clusters the per-sample mean
activations of the ESM-merged base model into k centroids per (task, layer), so
the router can match a new prompt's layer-input mean against the nearest of k
per-task centroids.

Output:
    prototypes[task_name][layer_name] = Tensor[k, d_in]   (k-means centroids)

Run:  python capture_moe_prototypes_km.py --base-model-path <merged> --k 8 \
          --num-samples 256 --output-path ./moe_prototypes_km8.pt
"""

import os
import gc
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from moe_config import TASK_CONFIGS, LINEAR_LAYER_NAMES
from capture_moe_layer_stats_ponly import formatting_prompts_func_input_only


def kmeans(X: torch.Tensor, k: int, iters: int = 20, seed: int = 0, device='cpu') -> torch.Tensor:
    """Plain Lloyd k-means (k-means++ init) on X [N, d] -> centroids [k, d].

    Runs on *device* (GPU) — the O(N*k*d) cdist is far faster there than on CPU.
    """
    X = X.to(device)
    N = X.shape[0]
    k = min(k, N)
    g = torch.Generator(device=device).manual_seed(seed)
    # k-means++ init
    idx = torch.randint(N, (1,), generator=g, device=device)
    C = [X[idx[0]]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / d2.sum()
        j = torch.multinomial(probs, 1, generator=g)
        C.append(X[j[0]])
        d2 = torch.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = torch.stack(C, 0)                              # [k, d]
    for _ in range(iters):
        dist = torch.cdist(X, C)                       # [N, k]
        assign = dist.argmin(1)
        newC = C.clone()
        for j in range(k):
            m = assign == j
            if m.any():
                newC[j] = X[m].mean(0)
        if torch.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return C.cpu()


@torch.no_grad()
def capture(base_model_path, output_path, k, num_samples, batch_size, device='cuda'):
    print(f'Loading merged base model from {base_model_path}')
    model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    num_layers = model.config.num_hidden_layers
    prototypes = {}

    for task_name, task_cfg in TASK_CONFIGS.items():
        print(f'\n=== task: {task_name} ===')
        dataset = load_dataset(task_cfg['hf_dataset'], split='train')
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
        texts = [formatting_prompts_func_input_only(ex, **task_cfg['format_keys']) for ex in dataset]

        batches = []
        for i in range(0, num_samples, batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=2048, return_tensors='pt')
            batches.append(enc)

        current_mask = [None]
        means = {}

        def make_hook(key):
            def hook(module, args, output):
                x = args[0].detach()                       # [B, seq, d_in]
                mb = current_mask[0].bool().unsqueeze(-1)  # [B, seq, 1]
                counts = current_mask[0].bool().sum(1).clamp(min=1).unsqueeze(-1)
                x_mean = (x * mb).sum(1) / counts          # [B, d_in]
                means.setdefault(key, []).append(x_mean.float().cpu())
            return hook

        handles = []
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                key = (layer_idx, short_name)
                module = model.model.layers[layer_idx]
                for attr in short_name.split('.'):
                    module = getattr(module, attr)
                handles.append(module.register_forward_hook(make_hook(key)))

        for i, enc in enumerate(batches):
            ids = enc['input_ids'].to(device)
            attn = enc['attention_mask'].to(device)
            current_mask[0] = attn
            _ = model(input_ids=ids, attention_mask=attn)
            del ids, attn
            torch.cuda.empty_cache()
            print(f'  batch {i+1}/{len(batches)}')

        for h in handles:
            h.remove()

        task_proto = {}
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                key = (layer_idx, short_name)
                layer_name = f'model.layers.{layer_idx}.{short_name}.weight'
                X = torch.cat(means[key], 0)               # [N, d_in]
                centroids = kmeans(X, k, device=device)    # [k, d_in]
                task_proto[layer_name] = centroids.contiguous()
                del X, means[key]
        prototypes[task_name] = task_proto
        del means, task_proto
        gc.collect()
        torch.cuda.empty_cache()
        print(f'  {task_name}: {len(prototypes[task_name])} layers, k={k}')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(prototypes, output_path)
    print(f'\nSaved k={k} prototypes to {output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-model-path', type=str, required=True)
    p.add_argument('--output-path', type=str, default='./moe_prototypes_km8.pt')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--num-samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--gpu-id', type=int, default=0)
    args = p.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu_id))
    capture(args.base_model_path, args.output_path, args.k, args.num_samples, args.batch_size)
