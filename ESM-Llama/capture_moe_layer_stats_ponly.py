"""
Capture MoE layer statistics (prompt-only, merged-model activations).

Differences from capture_moe_layer_stats.py:
  - ONE forward pass through the BASE (ESM-merged) model instead of two passes
    through the finetuned model.
  - BOTH the routing prototypes and the expert principal directions are computed
    from that single pass, on PROMPT-ONLY text (no response tokens).

Rationale: at inference the router runs on the merged/deployment model over the
prompt; so both the prototype (mean input activation) and the offset subspace
(x @ dW^T, dW = W_ft - W_base) should come from the same prompt-only, merged-model
activation distribution.

Output structure:
    moe_layer_stats[task_name][layer_name] = {
        'offset_eigenvalues': [d_out],
        'offset_eigenvectors': [d_out, d_out],
        'n_tokens': int,
    }
    prototypes[task_name][layer_name] = [d_in]   # mean input vector (prompt-only)
"""

import os
import gc
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from moe_config import (
    MODEL_LOCAL,
    TASK_CONFIGS,
    LINEAR_LAYER_NAMES,
    MoEConfig,
)


def formatting_prompts_func_input_only(examples, instruction_key='instruction',
                                       input_key='input', output_key=None):
    """Format prompt with ONLY the input part (no response)."""
    instruction = examples[instruction_key]
    input_text = examples.get(input_key, '')

    if input_text:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
'''
    else:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''
    return text


def compute_pca_from_gram(gram_matrix, n_samples, device='cpu'):
    """Compute PCA (eigenvalues, eigenvectors) from an accumulated Gram matrix."""
    d = gram_matrix.shape[0]
    gram_f32 = gram_matrix.to(torch.float32)

    if n_samples <= 1:
        return torch.zeros(d, dtype=torch.float32), torch.eye(d, dtype=torch.float32)

    gram_device = gram_f32.to(device)
    cov = gram_device / (n_samples - 1)

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    except Exception:
        reg = 1e-10 * torch.eye(d, dtype=torch.float32, device=cov.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov + reg)

    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    return eigenvalues.cpu(), eigenvectors.cpu()


@torch.no_grad()
def capture_moe_stats_and_prototypes_ponly(config: MoEConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    base_model_path = config.get_base_model_path()
    print(f'Loading base (merged) model from {base_model_path}')

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.bfloat16)
    base_model = base_model.to(device)
    base_model.eval()
    base_model.config.use_cache = False
    base_sd = base_model.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = base_model.config.num_hidden_layers
    print(f'Model has {num_layers} layers')

    moe_layer_stats = {}
    prototypes = {}

    for task_name, task_cfg in TASK_CONFIGS.items():
        print(f'\n{"="*60}')
        print(f'Processing task: {task_name}')
        print(f'{"="*60}')

        # Fine-tuned weights (CPU) — used ONLY to form dW = W_ft - W_base.
        ft_model_path = task_cfg['ft_model']
        print(f'Loading fine-tuned weights (CPU): {ft_model_path}')
        ft_model = AutoModelForCausalLM.from_pretrained(
            ft_model_path, dtype=torch.bfloat16, device_map='cpu',
        )
        ft_sd = ft_model.state_dict()

        dataset = load_dataset(task_cfg['hf_dataset'], split='train')
        dataset = dataset.shuffle(seed=42).select(range(config.num_samples_capture))
        format_keys = task_cfg['format_keys']

        texts = [formatting_prompts_func_input_only(ex, **format_keys) for ex in dataset]
        tokenizer.padding_side = 'right'

        batches = []
        for i in range(0, config.num_samples_capture, config.batch_size_capture):
            batch_texts = texts[i:i + config.batch_size_capture]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=2048, return_tensors='pt',
            )
            batches.append(enc)

        print(f'  {config.num_samples_capture} prompt-only samples, {len(batches)} micro-batches')

        accum_prototype = {}
        accum_offset = {}
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                weight_name = f'model.layers.{layer_idx}.{short_name}.weight'
                w = base_sd[weight_name]
                d_out, d_in = w.shape
                accum_prototype[(layer_idx, short_name)] = {
                    'input_sum': torch.zeros(d_in, dtype=torch.float32),
                    'n_tokens': 0,
                }
                accum_offset[(layer_idx, short_name)] = {
                    'offset_gram': torch.zeros(d_out, d_out, dtype=torch.float32),
                    'n_tokens': 0,
                }

        current_attention_mask = [None]

        def make_hook(key):
            layer_idx, short_name = key
            weight_name = f'model.layers.{layer_idx}.{short_name}.weight'
            ft_weight_cpu = ft_sd[weight_name]

            def hook(module, args, output):
                x = args[0].detach()
                mask = current_attention_mask[0]
                orig_shape = x.shape
                d_in = orig_shape[-1]

                x_flat = x.reshape(-1, d_in)
                mask_flat = mask.reshape(-1).bool()
                x_flat = x_flat[mask_flat]
                n_total = x_flat.shape[0]

                x_flat_f32 = x_flat.to(torch.float32)

                # Prototype: mean input activation (prompt-only tokens)
                accum_prototype[key]['input_sum'] += x_flat_f32.sum(dim=0).cpu()
                accum_prototype[key]['n_tokens'] += n_total

                # Offset Gram: dW = W_ft - W_base(merged)
                delta_W = (ft_weight_cpu.to(module.weight.device, torch.float32)
                           - module.weight.float())

                for start in range(0, n_total, config.chunk_size_capture):
                    end = min(start + config.chunk_size_capture, n_total)
                    x_chunk = x_flat_f32[start:end]

                    offset = x_chunk @ delta_W.T
                    chunk_offset_gram = (offset.T @ offset).cpu()
                    chunk_size = x_chunk.shape[0]
                    del offset, x_chunk

                    accum_offset[key]['offset_gram'] += chunk_offset_gram
                    accum_offset[key]['n_tokens'] += chunk_size
                    del chunk_offset_gram

                del x_flat_f32, delta_W

            return hook

        handles = []
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                key = (layer_idx, short_name)
                layer_module = base_model.model.layers[layer_idx]
                module = layer_module
                for attr in short_name.split('.'):
                    module = getattr(module, attr)
                handle = module.register_forward_hook(make_hook(key))
                handles.append(handle)

        print(f'  Running forward pass on merged model (prompt-only)...')
        for i, enc in enumerate(batches):
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            current_attention_mask[0] = attention_mask
            _ = base_model(input_ids=input_ids, attention_mask=attention_mask)
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            print(f'    processed {i+1}/{len(batches)} batches')

        for h in handles:
            h.remove()
        del handles, batches
        torch.cuda.empty_cache()
        del ft_model, ft_sd
        gc.collect()

        task_layer_stats = {}
        task_prototypes = {}
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                layer_name = f'model.layers.{layer_idx}.{short_name}.weight'

                proto_data = accum_prototype[(layer_idx, short_name)]
                n_proto = proto_data['n_tokens']
                if n_proto > 0:
                    prototype = proto_data['input_sum'] / n_proto
                else:
                    prototype = torch.zeros(base_sd[layer_name].shape[1], dtype=torch.float32)
                task_prototypes[layer_name] = prototype

                offset_data = accum_offset[(layer_idx, short_name)]
                n_offset = offset_data['n_tokens']
                offset_eigvals, offset_eigvecs = compute_pca_from_gram(offset_data['offset_gram'], n_offset, device)

                task_layer_stats[layer_name] = {
                    'offset_eigenvalues': offset_eigvals,
                    'offset_eigenvectors': offset_eigvecs,
                    'n_tokens': n_offset,
                }

                accum_prototype[(layer_idx, short_name)] = None
                accum_offset[(layer_idx, short_name)] = None
                if (layer_idx * len(LINEAR_LAYER_NAMES) + LINEAR_LAYER_NAMES.index(short_name)) % 28 == 0:
                    gc.collect()

                denom = offset_eigvals.sum().item()
                offset_var = offset_eigvals[:5].sum().item() / denom * 100 if denom > 0 else 0.0
                print(f'  {layer_name}: offset PC5={offset_var:.1f}%, prototype_tokens={n_proto}')

        moe_layer_stats[task_name] = task_layer_stats
        prototypes[task_name] = task_prototypes

        del accum_prototype, accum_offset, task_layer_stats, task_prototypes
        gc.collect()
        torch.cuda.empty_cache()

    del base_model, base_sd
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(moe_layer_stats, config.layer_stats_path)
    print(f'\nSaved MoE layer stats to {config.layer_stats_path}')
    torch.save(prototypes, config.prototype_path)
    print(f'Saved MoE prototypes to {config.prototype_path}')

    return moe_layer_stats, prototypes


def main():
    parser = argparse.ArgumentParser(description='Capture MoE layer stats (prompt-only, merged-model)')
    parser.add_argument('--base-model-path', type=str, default='',
                        help='Path to the ESM-merged base model')
    parser.add_argument('--layer-stats-path', type=str, default='./moe_layer_stats_ponly.pt')
    parser.add_argument('--prototype-path', type=str, default='./moe_prototypes_ponly.pt')
    parser.add_argument('--num-samples', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args()

    config = MoEConfig(
        base_model_path=args.base_model_path,
        layer_stats_path=args.layer_stats_path,
        prototype_path=args.prototype_path,
        num_samples_capture=args.num_samples,
        batch_size_capture=args.batch_size,
        chunk_size_capture=args.chunk_size,
    )
    print(config)

    capture_moe_stats_and_prototypes_ponly(config)


if __name__ == '__main__':
    main()
