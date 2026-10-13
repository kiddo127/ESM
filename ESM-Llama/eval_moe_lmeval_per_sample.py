"""
lm_eval evaluation for MoE model with PER-SAMPLE routing.

Compared to eval_moe_lmeval.py (batch-level routing), this version routes each
sample in a batch to its own best expert independently.

Usage (same CLI as eval_moe_lmeval.py)::

    python eval_moe_lmeval_per_sample.py \\
        --base-model-path ./merged_models/... \\
        --layer-stats-path ./moe_layer_stats.pt \\
        --prototype-path ./moe_prototypes.pt \\
        --tasks gsm8k_cot --batch-size 16 \\
        --output-path ./results/gsm8k
"""

import os
import json
import time
import argparse
import torch

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# If the hub requires authentication, export HF_HUB_TOKEN in your environment.

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from moe_config import MoEConfig
from moe_model_for_eval import create_moe_model_for_eval
from moe_per_sample_router import PerSampleRouter


class PerSampleMoEHFLM(HFLM):
    """HFLM subclass with per-sample MoE routing (fix_agg route-then-fix).

    fix_agg: route with cross-layer aggregation (one expert per sample), merge it,
    infer, restore. The router is built with aggregate_routing=True. The whole
    batch is routed in ONE forward (routing is per-sample independent), then
    samples are grouped by expert and each group is merged once + generated
    together — so a same-expert batch (e.g. gsm8k->math) is one merge + one
    batched generation. The selected experts are folded into the weights once per
    sample, so generation runs with zero routing/LoRA overhead.

    Oracle / base modes work identically to the original MoEHFLM.
    """

    def __init__(self, moe_model, router: PerSampleRouter, **kwargs):
        self.moe_model = moe_model
        self.router = router
        self.timing = {'route_s': 0.0, 'merge_s': 0.0, 'restore_s': 0.0,
                       'infer_s': 0.0, 'calls': 0}
        super().__init__(pretrained=moe_model.base_model, **kwargs)

    def _enter_routing(self, inps, attn_mask, use_cache):
        """Route the prompt, then merge+fix the model.

        Returns a cleanup callable that restores the base weights.
        """
        if attn_mask is None:
            attn_mask = (inps != self.tokenizer.pad_token_id).long()
        t0 = time.time()
        self.router.compute_per_sample_routing(inps, attn_mask, use_cache=use_cache)
        self.timing['route_s'] += time.time() - t0

        dominant = self.router.get_dominant_expert()
        if dominant:
            self.moe_model.apply_expert_nonlinear(dominant)

        t0 = time.time()
        self.moe_model.apply_per_sample_routing(self.router.per_sample_routing)
        self.moe_model.merge_expert_weights()
        self.timing['merge_s'] += time.time() - t0

        def cleanup():
            t0 = time.time()
            self.moe_model.restore_base_weights()
            self.timing['restore_s'] += time.time() - t0
        return cleanup

    def _fix_agg_batches(self, inps, attn_mask, use_cache):
        """Batched route-then-fix: route the whole batch once, group samples by
        their dominant expert, and return [(sample_indices, task), ...].

        Routing is per-sample independent, so routing the batch is identical to
        routing each sample separately; the win is that the ~590 Python hook
        dispatches per sample are amortised across the batch.
        """
        if attn_mask is None:
            attn_mask = (inps != self.tokenizer.pad_token_id).long()
        t0 = time.time()
        self.router.compute_per_sample_routing(inps, attn_mask, use_cache=use_cache)
        self.timing['route_s'] += time.time() - t0

        dominant = self.router.get_dominant_expert_per_sample()
        groups = {}
        for i, task in enumerate(dominant):
            groups.setdefault(task, []).append(i)
        return groups

    def _merge_for_group(self, indices):
        """Merge the (single per-sample) expert of a group into the weights.

        For fix_agg every layer of every sample in the group uses the same
        aggregated expert, so apply_per_sample_routing's sample-0 semantics is
        exact for the whole group.
        """
        per = self.router.per_sample_routing
        sub_routing = {ln: t[indices] for ln, t in per.items()}
        self.moe_model.apply_per_sample_routing(sub_routing)
        t0 = time.time()
        self.moe_model.merge_expert_weights()
        self.timing['merge_s'] += time.time() - t0

    def _restore(self):
        t0 = time.time()
        self.moe_model.restore_base_weights()
        self.timing['restore_s'] += time.time() - t0

    def _model_call(self, inps, attn_mask=None, labels=None):
        if self.moe_model.routing_mode != 'auto':
            if attn_mask is not None and labels is not None:
                return super()._model_call(inps, attn_mask=attn_mask, labels=labels)
            return super()._model_call(inps)

        # Batched route-then-fix (fix_agg): one routing forward per batch.
        if inps.shape[0] > 1:
            groups = self._fix_agg_batches(inps, attn_mask, use_cache=True)
            self.timing['calls'] += inps.shape[0]
            outs = [None] * inps.shape[0]
            for task, idxs in groups.items():
                self.moe_model.apply_expert_nonlinear(task)
                self._merge_for_group(idxs)
                sub = inps[idxs]
                t0 = time.time()
                try:
                    if attn_mask is not None and labels is not None:
                        out = super()._model_call(sub, attn_mask=attn_mask[idxs], labels=labels[idxs])
                    else:
                        out = super()._model_call(sub)
                finally:
                    self.timing['infer_s'] += time.time() - t0
                    self._restore()
                for j, i in enumerate(idxs):
                    outs[i] = out[j:j + 1]
            return torch.cat(outs, dim=0)

        cleanup = self._enter_routing(inps, attn_mask, use_cache=True)
        self.timing['calls'] += 1
        t0 = time.time()
        try:
            if attn_mask is not None and labels is not None:
                return super()._model_call(inps, attn_mask=attn_mask, labels=labels)
            return super()._model_call(inps)
        finally:
            self.timing['infer_s'] += time.time() - t0
            cleanup()

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        if self.moe_model.routing_mode != 'auto':
            return super()._model_generate(context, max_length, stop, **generation_kwargs)

        # Batched route-then-fix (fix_agg): one routing forward per batch.
        if context.shape[0] > 1:
            # lm_eval passes the full-batch attention_mask inside generation_kwargs;
            # it must be sliced alongside the context for each per-expert group.
            batch_attn = generation_kwargs.pop('attention_mask', None)
            groups = self._fix_agg_batches(context, batch_attn, use_cache=False)
            self.timing['calls'] += context.shape[0]
            outs = [None] * context.shape[0]
            for task, idxs in groups.items():
                self.moe_model.apply_expert_nonlinear(task)
                self._merge_for_group(idxs)
                sub = context[idxs]
                gw = dict(generation_kwargs)
                if batch_attn is not None:
                    gw['attention_mask'] = batch_attn[idxs]
                t0 = time.time()
                try:
                    out = super()._model_generate(sub, max_length, stop, **gw)
                finally:
                    self.timing['infer_s'] += time.time() - t0
                    self._restore()
                for j, i in enumerate(idxs):
                    outs[i] = out[j:j + 1]
            # Groups can stop at different lengths; right-pad to a common width.
            maxlen = max(o.shape[1] for o in outs)
            outs = [torch.nn.functional.pad(o, (0, maxlen - o.shape[1]),
                                            value=self.tokenizer.pad_token_id)
                    if o.shape[1] < maxlen else o for o in outs]
            return torch.cat(outs, dim=0)

        attention_mask = (context != self.tokenizer.pad_token_id).long()
        cleanup = self._enter_routing(context, attention_mask, use_cache=False)
        self.timing['calls'] += 1
        t0 = time.time()
        try:
            return super()._model_generate(context, max_length, stop, **generation_kwargs)
        finally:
            self.timing['infer_s'] += time.time() - t0
            cleanup()


def run_lm_eval_tasks(eval_wrapper, task_names, output_path, limit=None):
    results = evaluator.simple_evaluate(
        model=eval_wrapper,
        tasks=task_names,
        batch_size=eval_wrapper.batch_size,
        limit=limit,
    )

    os.makedirs(output_path, exist_ok=True)

    if results is not None:
        save_file = os.path.join(output_path, 'lm_eval_results.json')
        with open(save_file, 'w') as f:
            json.dump(results['results'], f, indent=2)

        print("\nResults:")
        for task_name, task_results in results['results'].items():
            if 'acc' in task_results:
                print(f"  {task_name}: acc={task_results['acc']:.4f}")
            elif 'acc_norm' in task_results:
                print(f"  {task_name}: acc_norm={task_results['acc_norm']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='lm_eval for MoE model (per-sample routing)')
    parser.add_argument('--base-model-path', type=str, default='')
    parser.add_argument('--layer-stats-path', type=str, default='./moe_layer_stats.pt')
    parser.add_argument('--prototype-path', type=str, default='./moe_prototypes_km8.pt')
    parser.add_argument('--moe-rank', type=int, default=64)
    parser.add_argument('--moe-scaling', type=float, default=1.0)
    parser.add_argument('--routed-layers', type=str, nargs='+', default=None,
                        help='Layer types to route, e.g. self_attn.q_proj mlp.gate_proj mlp.up_proj')
    parser.add_argument('--output-path', type=str, default='./results/moe_lmeval')
    parser.add_argument('--tasks', type=str, default='gsm8k_cot',
                        help='Comma-separated task names')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--oracle-task', type=str, default='',
                        help='Task name for oracle routing mode. Empty for auto routing.')
    parser.add_argument('--base-mode', action='store_true',
                        help='Use base model only without any expert LoRA.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples for quick testing.')
    parser.add_argument('--routing-mode', type=str, default='per_layer',
                        choices=['per_layer', 'per_block'],
                        help="Routing strategy: 'per_layer' (default) or 'per_block'")
    parser.add_argument('--moe-start-layer', type=int, default=0,
                        help='First layer index to apply MoE.')
    parser.add_argument('--aggregate-routing', action='store_true',
                        help='Aggregate routing scores across layers; each sample '
                             'gets one consistent expert for all layers.')
    parser.add_argument('--config-task', type=str, default='',
                        help='Use this expert\'s config and non-linear params (embed/lm_head/norms). '
                             'Useful for auto/base modes when evaluating a specific task.')
    parser.add_argument('--expected-expert', type=str, default='',
                        help='Track routing accuracy against this expected expert '
                             '(e.g. math for gsm8k, instruction for ifeval, etc.)')
    args = parser.parse_args()

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
    print("Loading MoE Model for lm_eval (per-sample routing)")
    print(f"{'='*60}")
    print(config)

    load_start = time.time()
    moe_model = create_moe_model_for_eval(config)

    if args.oracle_task:
        moe_model.set_oracle_mode(args.oracle_task)
        moe_model.merge_expert_into_model(args.oracle_task)
        # The expert is LoRA *plus* its non-linear params (embed/lm_head/norms);
        # without these the oracle is a handicapped expert.
        moe_model.apply_expert_nonlinear(args.oracle_task)
    elif args.base_mode:
        moe_model.set_base_mode()
        if args.config_task:
            moe_model.apply_expert_config(args.config_task)
            moe_model.apply_expert_nonlinear(args.config_task)
    # else: auto mode (default)
    if not args.oracle_task and not args.base_mode and args.config_task:
        moe_model.apply_expert_config(args.config_task)
        moe_model.apply_expert_nonlinear(args.config_task)

    # Set expected expert for routing accuracy tracking
    if args.expected_expert:
        moe_model.set_expected_expert(args.expected_expert)
        # Apply config once at start (no per-batch switching needed)
        if moe_model.routing_mode == 'auto':
            moe_model.apply_expert_config(args.expected_expert)

    # fix_agg folds the selected expert(s) into the weights: one expert per
    # sample; routing is per-sample independent, so the whole batch can be routed
    # in ONE forward (amortising the per-layer hook dispatch), then merged and
    # generated in per-expert groups. Batch size is kept.
    aggregate_routing = True

    # Create per-sample router (used only in auto mode)
    router = PerSampleRouter(moe_model, aggregate_routing=aggregate_routing) \
        if moe_model.routing_mode == 'auto' else None
    if router is not None and args.expected_expert:
        router.set_expected_expert(args.expected_expert)

    print(f"Model loaded in {time.time() - load_start:.1f}s")

    task_names = args.tasks.split(',')

    print(f"\n{'='*60}")
    print(f"Running lm_eval tasks: {task_names}")
    print(f"Mode: {moe_model.routing_mode.upper()}")
    if moe_model.routing_mode == 'auto':
        print("Routing: PER-SAMPLE (each sample routes independently)")
    print(f"{'='*60}")

    eval_wrapper = PerSampleMoEHFLM(
        moe_model,
        router,
        device='cuda',
        dtype='bfloat16',
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
    )

    eval_start = time.time()
    results = run_lm_eval_tasks(
        eval_wrapper, task_names, args.output_path, limit=args.limit,
    )
    eval_time = time.time() - eval_start

    print(f"\nEvaluation time: {eval_time:.1f}s")
    print(f"Results saved to: {args.output_path}")

    # Save timing breakdown (auto mode only)
    if moe_model.routing_mode == 'auto':
        t = eval_wrapper.timing
        t['eval_total_s'] = round(eval_time, 2)
        t['fix_mode'] = 'fix_agg'
        t['batch_size'] = args.batch_size
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, 'timing.json'), 'w') as f:
            json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}, f, indent=2)
        print(f"Timing: {t}")

    # Print routing accuracy
    if router is not None:
        router.print_routing_accuracy()
    moe_model.print_routing_accuracy()

    moe_model.cleanup()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
