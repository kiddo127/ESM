"""
Coding evaluation for MoE model with PER-SAMPLE routing.

Compared to eval_moe_bigcode.py (batch-level routing), this version routes each
sample independently.

Note: BigCode eval typically processes one prompt at a time (batch_size=1 for
routing) with num_return_sequences for multiple samples.  Per-sample routing
naturally handles the num_return_sequences expansion — all generated sequences
from the same prompt inherit the same expert.
"""

import os
import json
import time
import argparse

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# If the hub requires authentication, export HF_HUB_TOKEN in your environment.
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

import torch
from transformers import AutoTokenizer, StoppingCriteriaList
from tqdm import tqdm

from bigcode_eval.tasks import get_task
from bigcode_eval.generation import EndOfFunctionCriteria

from moe_config import MoEConfig
from moe_model_for_eval import MoEModelForCausalLM
from moe_per_sample_router import PerSampleRouter


def run_coding_eval_per_sample(model, tokenizer, task_name, output_dir, n_samples=10,
                                temperature=0.2, device='cuda:0', aggregate_routing=False,
                                expected_expert='', timing=None,
                                route_batch_size=8):
    """Run coding evaluation with per-sample routing (fix_agg route-then-fix).

    Routing is done per prompt; with route_batch_size > 1 the prompts are routed
    in chunks (one forward per chunk — routing is per-sample independent), which
    amortises the per-layer hook dispatch.  Generation stays per prompt (one
    prompt x n_samples), so the merge is still a single per-prompt expert.
    fix_agg routes on the prompt then MERGES the selected expert into the weights
    (fixed model).
    """
    if timing is None:
        timing = {'route_s': 0.0, 'merge_s': 0.0, 'restore_s': 0.0, 'infer_s': 0.0, 'calls': 0}
    task = get_task(task_name)
    dataset = task.get_dataset()
    print(f"  Dataset size: {len(dataset)}")

    generations_all = []
    references_all = []

    if tokenizer.eos_token:
        if task.stop_words:
            if tokenizer.eos_token not in task.stop_words:
                task.stop_words.append(tokenizer.eos_token)
        else:
            task.stop_words = [tokenizer.eos_token]
    stop_words = task.stop_words

    is_auto = (model.routing_mode == 'auto')
    router = PerSampleRouter(model, aggregate_routing=aggregate_routing) if is_auto else None
    if router is not None and expected_expert:
        router.set_expected_expert(expected_expert)

    # Pre-route all prompts in chunks.  Routing is per-sample independent, so a
    # chunked forward is equivalent to routing each prompt on its own, but the
    # per-layer hook dispatch is amortised across the chunk.
    precomputed = None
    if is_auto and router is not None and route_batch_size > 1:
        precomputed = [None] * len(dataset)
        for a in range(0, len(dataset), route_batch_size):
            chunk = list(range(a, min(a + route_batch_size, len(dataset))))
            enc = tokenizer([task.get_prompt(dataset[b]) for b in chunk],
                            return_tensors='pt', padding=True, truncation=True, max_length=512)
            t0 = time.time()
            router.compute_per_sample_routing(enc['input_ids'].to(device),
                                              enc['attention_mask'].to(device))
            timing['route_s'] += time.time() - t0
            doms = router.get_dominant_expert_per_sample()
            for local, b in enumerate(chunk):
                precomputed[b] = (
                    doms[local],
                    {ln: t[local:local + 1].clone() for ln, t in router.per_sample_routing.items()},
                )
        print(f"  Pre-routed {len(dataset)} prompts in chunks of {route_batch_size}")

    for i, ex in enumerate(tqdm(dataset, desc=task_name)):
        prompt = task.get_prompt(ex)

        input_ids = tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=512
        )['input_ids'].to(device)

        # Route the prompt, then either enable hooks or merge+fix the model.
        if is_auto and router is not None:
            if precomputed is not None:
                dominant, sub_routing = precomputed[i]
                router._per_sample_routing = sub_routing
            else:
                t0 = time.time()
                router.compute_per_sample_routing(input_ids, None)
                timing['route_s'] += time.time() - t0
                dominant = router.get_dominant_expert()
                sub_routing = router.per_sample_routing
            if dominant:
                model.apply_expert_nonlinear(dominant)
            t0 = time.time()
            model.apply_per_sample_routing(sub_routing)
            model.merge_expert_weights()
            timing['merge_s'] += time.time() - t0

        start_length = input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([
            EndOfFunctionCriteria(start_length, stop_words, tokenizer)
        ])

        t0 = time.time()
        with torch.no_grad():
            outputs = model.base_model.generate(
                input_ids=input_ids,
                max_length=512,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=n_samples,
                stopping_criteria=stopping_criteria,
            )
        timing['infer_s'] += time.time() - t0
        timing['calls'] += 1

        if is_auto and router is not None:
            t0 = time.time()
            model.restore_base_weights()
            timing['restore_s'] += time.time() - t0

        samples = []
        for j in range(n_samples):
            raw = outputs[j]
            if tokenizer.bos_token_id is not None and raw[0].item() == tokenizer.bos_token_id:
                raw = raw[1:]
            full_output = tokenizer.decode(raw, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            postprocessed = task.postprocess_generation(full_output, i)
            samples.append(postprocessed)

        generations_all.append(samples)
        ref = task.get_reference(ex)
        references_all.append(ref)

    # Save completions
    completions = []
    for i, ex in enumerate(tqdm(dataset, desc="Saving")):
        task_id = ex.get('task_id', str(i))
        completions.append({'task_id': task_id, 'completion': generations_all[i]})
    completions_file = os.path.join(output_dir, f'{task_name}_completions.json')
    with open(completions_file, 'w') as f:
        json.dump(completions, f, indent=2)

    print(f"  Computing pass@k...")
    results = task.process_results(generations_all, references_all)
    print(f"  Results: {results}")

    if router is not None:
        router.cleanup()
    return results


def main():
    parser = argparse.ArgumentParser(description='Coding evaluation for MoE (per-sample routing)')
    parser.add_argument('--base-model-path', type=str, default='')
    parser.add_argument('--layer-stats-path', type=str, default='./moe_layer_stats.pt')
    parser.add_argument('--prototype-path', type=str, default='./moe_prototypes_km8.pt')
    parser.add_argument('--moe-rank', type=int, default=64)
    parser.add_argument('--moe-scaling', type=float, default=1.0)
    parser.add_argument('--routed-layers', type=str, nargs='+', default=None,
                        help='Layer types to route, e.g. self_attn.q_proj mlp.gate_proj mlp.up_proj')
    parser.add_argument('--output-path', type=str, default='./results/moe_eval/code_eval.json')
    parser.add_argument('--tasks', type=str, default='humanevalplus,mbppplus')
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--oracle-task', type=str, default='')
    parser.add_argument('--base-mode', action='store_true',
                        help='Use base model only without any expert LoRA.')
    parser.add_argument('--routing-mode', type=str, default='per_layer',
                        choices=['per_layer', 'per_block'],
                        help="Routing strategy: 'per_layer' (default) or 'per_block'")
    parser.add_argument('--moe-start-layer', type=int, default=0,
                        help='First layer index to apply MoE.')
    parser.add_argument('--aggregate-routing', action='store_true',
                        help='Aggregate routing scores across layers; each sample '
                             'gets one consistent expert for all layers.')
    parser.add_argument('--expected-expert', type=str, default='',
                        help='Track routing accuracy against this expected expert.')
    parser.add_argument('--route-batch-size', type=int, default=8,
                        help='Number of prompts routed per forward (routing only; '
                             'generation stays per prompt). 1 = route per prompt.')
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    config = MoEConfig(
        base_model_path=args.base_model_path,
        moe_rank=args.moe_rank,
        moe_scaling=args.moe_scaling,
        layer_stats_path=args.layer_stats_path,
        prototype_path=args.prototype_path,
        routed_layers=args.routed_layers,
        routing_mode=args.routing_mode,
        moe_start_layer=args.moe_start_layer,
    )

    print(f"\n{'='*60}")
    print("Loading MoE Model for Coding Evaluation (per-sample routing)")
    print(f"{'='*60}")
    print(config)

    load_start = time.time()
    moe_model = MoEModelForCausalLM(config)

    if args.oracle_task:
        moe_model.set_oracle_mode(args.oracle_task)
        moe_model.merge_expert_into_model(args.oracle_task)
        moe_model.apply_expert_nonlinear(args.oracle_task)
    if args.base_mode:
        print("Base mode: using model without expert LoRA")
        moe_model.set_base_mode()
    if args.expected_expert:
        moe_model.set_expected_expert(args.expected_expert)
        if moe_model.routing_mode == 'auto':
            moe_model.apply_expert_config(args.expected_expert)
        # Also set on router (created per-sample in the eval loop)

    tokenizer = AutoTokenizer.from_pretrained(config.get_base_model_path())
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left'

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    output_dir = os.path.dirname(args.output_path)
    if not output_dir:
        output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    eval_start = time.time()

    all_metrics = {}
    task_list = args.tasks.split(',')
    timing = {'route_s': 0.0, 'merge_s': 0.0, 'restore_s': 0.0, 'infer_s': 0.0, 'calls': 0}
    effective_aggregate = True

    for task_name in task_list:
        metrics = run_coding_eval_per_sample(
            moe_model, tokenizer, task_name, output_dir,
            n_samples=args.n_samples, temperature=args.temperature, device=device,
            aggregate_routing=effective_aggregate,
            expected_expert=args.expected_expert,
            timing=timing,
            route_batch_size=args.route_batch_size,
        )
        all_metrics[task_name] = metrics

    all_metrics['config'] = {
        'model': args.base_model_path,
        'n_samples': args.n_samples,
        'temperature': args.temperature,
        'moe_rank': args.moe_rank,
        'moe_scaling': args.moe_scaling,
        'oracle_task': args.oracle_task or 'auto',
        'routing': 'per_sample',
        'fix_mode': 'fix_agg',
    }

    output_path = args.output_path.replace('.json', '_per_sample.json')
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    eval_time = time.time() - eval_start
    print(f"\nCoding evaluation time: {eval_time:.1f}s")
    print(f"Results saved to: {output_path}")

    if moe_model.routing_mode == 'auto':
        timing['eval_total_s'] = round(eval_time, 2)
        timing['fix_mode'] = 'fix_agg'
        with open(os.path.join(output_dir, 'timing.json'), 'w') as f:
            json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in timing.items()}, f, indent=2)
        print(f"Timing: {timing}")
    print("\nResults:")
    for task, metrics in all_metrics.items():
        if task != 'config':
            p1 = metrics.get('pass@1', 'N/A')
            p10 = metrics.get('pass@10', 'N/A')
            print(f"  {task}: pass@1={p1}, pass@10={p10}")

    moe_model.cleanup()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
