"""
Compute and save output offset PCA (principal directions) for all linear layers
across 5 tasks. ESM only consumes the offset eigenvectors, so the input Gram
matrix (X^T X) is not stored (it is large and unused).

Structure of output dict:
    layer_stats[task_name][layer_name] = {
        'offset_eigenvalues': Tensor [d_out],       # PCA eigenvalues of output offsets
        'offset_eigenvectors': Tensor [d_out, d_out],  # PCA eigenvectors (columns)
        'n_tokens': int,
    }
"""

import os
import gc
import torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

MODEL_BASE = 'meta-llama/Llama-3.2-3B'
MODEL_LOCAL = os.environ.get('BASE_MODEL_PATH', MODEL_BASE)

NUM_SAMPLES = 64
BATCH_SIZE = 16
CHUNK_SIZE = 4096

TASK_CONFIGS = {
    'instruction': {
        'hf_dataset': 'MergeBench/instruction_val',
        'format_keys': {'instruction_key': 'instruction', 'output_key': 'output'},
    },
    'math': {
        'hf_dataset': 'MergeBench/math_val',
        'format_keys': {'instruction_key': 'query', 'output_key': 'response'},
    },
    'coding': {
        'hf_dataset': 'MergeBench/coding_val',
        'format_keys': {'instruction_key': 'instruction', 'output_key': 'response'},
    },
    'safety': {
        'hf_dataset': 'MergeBench/safety_val',
        'format_keys': {'instruction_key': 'prompt', 'output_key': 'response'},
    },
    'multilingual': {
        'hf_dataset': 'MergeBench/multilingual_val',
        'format_keys': {'instruction_key': 'inputs', 'output_key': 'targets'},
    },
}

LINEAR_LAYER_NAMES = [
    'self_attn.q_proj',
    'self_attn.k_proj',
    'self_attn.v_proj',
    'self_attn.o_proj',
    'mlp.gate_proj',
    'mlp.up_proj',
    'mlp.down_proj',
]


def formatting_prompts_func(examples, instruction_key='instruction', input_key='input', output_key='output'):
    instruction = examples[instruction_key]
    input_text = examples.get(input_key, '')
    response = examples[output_key]

    if input_text:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input_text}

        ### Response:
        {response}
        '''
    else:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        {response}
        '''
    return text


def get_ft_model_path(base_model_name, task_name):
    model_name = base_model_name.split('/')[-1]
    return f'MergeBench/{model_name}_{task_name}'


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
    except:
        reg = 1e-10 * torch.eye(d, dtype=torch.float32, device=cov.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov + reg)

    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    return eigenvalues.cpu(), eigenvectors.cpu()


@torch.no_grad()
def capture_layer_stats():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load base model to CPU only
    print(f'Loading base model from {MODEL_LOCAL}')
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL, dtype=torch.bfloat16)
    base_model.eval()
    base_sd = base_model.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = base_model.config.num_hidden_layers

    # Result dict: task_name -> layer_name -> {gram + PCA}
    layer_stats = {}

    for task_name, task_cfg in TASK_CONFIGS.items():
        print(f'\n{"="*60}')
        print(f'Processing task: {task_name}')
        print(f'{"="*60}')

        # 1. Load fine-tuned model
        ft_model_path = get_ft_model_path(MODEL_BASE, task_name)
        print(f'Loading fine-tuned model: {ft_model_path}')
        ft_model = AutoModelForCausalLM.from_pretrained(
            ft_model_path, dtype=torch.bfloat16, device_map=device,
        )
        ft_model.eval()
        ft_model.config.use_cache = False

        # 2. Load and tokenize dataset into micro-batches
        dataset = load_dataset(task_cfg['hf_dataset'], split='train')
        # print(task_cfg['hf_dataset'], len(dataset)) # 1000
        dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
        format_keys = task_cfg['format_keys']
        texts = [formatting_prompts_func(ex, **format_keys) for ex in dataset]
        tokenizer.padding_side = 'right'

        sample_batches = []
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=2048, return_tensors='pt',
            )
            sample_batches.append(enc)
        print(f'  {NUM_SAMPLES} samples, {len(sample_batches)} micro-batches (batch_size={BATCH_SIZE})')

        # 3. Initialize accumulators: store offset Gram for each layer
        #    key -> { 'offset_gram': Tensor [d_out, d_out], 'n_tokens': int }
        gram_accum = {}
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                weight_name = f'model.layers.{layer_idx}.{short_name}.weight'
                w = base_sd[weight_name]
                d_out = w.shape[0]
                gram_accum[(layer_idx, short_name)] = {
                    'offset_gram': torch.zeros(d_out, d_out, dtype=torch.float32),
                    'n_tokens': 0,
                }

        # 4. Register hooks
        current_attention_mask = [None]

        def make_hook(key):
            layer_idx, short_name = key
            weight_name = f'model.layers.{layer_idx}.{short_name}.weight'
            base_weight = base_sd[weight_name]

            def hook(module, args, output):
                x = args[0].detach()  # [batch, seq, d_in], on GPU
                mask = current_attention_mask[0]  # [batch, seq]
                orig_shape = x.shape
                x_flat = x.reshape(-1, orig_shape[-1])  # [batch*seq, d_in]
                mask_flat = mask.reshape(-1).bool()  # [batch*seq]
                x_flat = x_flat[mask_flat]  # keep only non-padding tokens
                n_total = x_flat.shape[0]

                # Cache delta_W on GPU in float32 after first call
                # Compute delta_W on-the-fly in float32, then free
                delta_W = (module.weight - base_weight.to(module.weight.device, dtype=module.weight.dtype)).to(torch.float32)

                d_out, d_in = delta_W.shape

                x_flat_f32 = x_flat.to(torch.float32)
                del x_flat

                for start in range(0, n_total, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_total)
                    x_chunk = x_flat_f32[start:end]

                    offset = x_chunk @ delta_W.T
                    chunk_offset_gram = (offset.T @ offset).cpu()
                    chunk_size = x_chunk.shape[0]
                    del offset, x_chunk

                    gram_accum[key]['offset_gram'] += chunk_offset_gram
                    gram_accum[key]['n_tokens'] += chunk_size
                    del chunk_offset_gram

                del x_flat_f32, delta_W
            return hook

        handles = []
        for layer_idx in range(num_layers):
            for short_name in LINEAR_LAYER_NAMES:
                key = (layer_idx, short_name)
                layer_module = ft_model.model.layers[layer_idx]
                module = layer_module
                for attr in short_name.split('.'):
                    module = getattr(module, attr)
                # print(module)
                handle = module.register_forward_hook(make_hook(key))
                handles.append(handle)
        # exit()

        # 5. Forward pass
        num_batches = len(sample_batches)
        print(f'  Running forward pass ({num_batches} micro-batches)...')
        for i, enc in enumerate(sample_batches):
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            current_attention_mask[0] = attention_mask
            _ = ft_model(input_ids=input_ids, attention_mask=attention_mask)
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            print(f'    processed {i+1}/{num_batches} micro-batches')

        # 6. Remove hooks and free ft_model
        for h in handles:
            h.remove()
        del handles, sample_batches
        torch.cuda.empty_cache()
        del ft_model
        torch.cuda.empty_cache()
        gc.collect()

        # 7. Compute PCA on both Gram matrices, saving results
        print(f'  Computing PCA and saving results...')
        task_results = {}
        for key, accum in gram_accum.items():
            layer_idx, short_name = key
            layer_name = f'model.layers.{layer_idx}.{short_name}.weight'
            n = accum['n_tokens']

            # PCA on output offsets only
            offset_eigvals, offset_eigvecs = compute_pca_from_gram(accum['offset_gram'], n, device)

            task_results[layer_name] = {
                'offset_eigenvalues': offset_eigvals,
                'offset_eigenvectors': offset_eigvecs,
                'n_tokens': n,
            }

            # Free this entry immediately
            gram_accum[key] = None
            if (layer_idx * len(LINEAR_LAYER_NAMES) + LINEAR_LAYER_NAMES.index(short_name)) % 28 == 0:
                gc.collect()

            offset_var = offset_eigvals[:5].sum().item() / offset_eigvals.sum().item() * 100
            print(f'  {layer_name}: offset PC5={offset_var:.1f}%')

        layer_stats[task_name] = task_results

        del gram_accum, task_results
        gc.collect()
        torch.cuda.empty_cache()

    del base_model, base_sd
    gc.collect()
    torch.cuda.empty_cache()

    print(f'\n{"="*60}')
    print('Done! Summary:')
    print(f'{"="*60}')
    for task_name, layers in layer_stats.items():
        print(f'\n{task_name}: {len(layers)} layers')
        for layer_name, info in list(layers.items())[:2]:
            print(f'  {layer_name}:')
            print(f'    offset_eigenvectors: {list(info["offset_eigenvectors"].shape)}  n_tokens={info["n_tokens"]}')

    # Save to file
    save_path = './layer_stats.pt'
    torch.save(layer_stats, save_path)
    print(f'\nSaved layer stats to {save_path}')

    return layer_stats


if __name__ == '__main__':
    layer_stats = capture_layer_stats()