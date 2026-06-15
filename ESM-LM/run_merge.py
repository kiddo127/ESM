import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from collections import defaultdict
import tqdm
import re
import utils
from transformers import AutoTokenizer
from param import param
import numpy as np
from merge import MergingMethod
import eval
import inspect
import datasets
import pandas as pd

from essential_subspace_decomposition import get_principal_direction
from search_scaling import search_scaling_for_merge
from esm_moe_eval import ESMMoEEvaluator

args = None
DEVICE = 'cuda:0'
BASE_DATA_DIR = '/seu_share2/home/gengxin/230238562/WorkBench/ESM-LM/discriminative_moe'



expert_performance = {
    'base':{
        'cola': 56.52,
        'sst2': 94.72,
        'mrpc': 87.99,
        'stsb': 86.36,
        'qqp': 89.71,
        'mnli': 87.01,
        'qnli': 91.71,
        'rte': 66.43,
    },
    'large':{
        'cola': 64.11,
        'sst2': 95.99,
        'mrpc': 87.87,
        'stsb': 90.33,
        'qqp': 90.35,
        'mnli': 90.41,
        'qnli': 94.21,
        'rte': 75.81,
    }
}

@torch.inference_mode()
def run_pretrained(
    args,
    load_head=True,
): 

    global model_path_template, head_path_template
    model_path_template='../roberta/{name}/roberta-base_lr1e-05'
    head_path_template='../roberta/{name}/roberta-base_lr1e-05/classifier_head.pt'
    if 'large' in args.base_model:
        model_path_template='../roberta_large/{name}/roberta-large_lr1e-05'
        head_path_template='../roberta_large/{name}/roberta-large_lr1e-05/classifier_head.pt'

    # \theta_t
    pretrained = utils.load_classifier(args.base_model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    data = utils.from_json(args.data_path)
    metrics = {'model': args.base_model }
    dataset_list = defaultdict(list)
    for data_item in (data):
        data_id = data_item['dataset_ids']
        data_name = list(eval.glue_data_id_map.keys())[data_id]
        dataset_list[data_name].append(data_item)

    for data_name, dataset in dataset_list.items():

        dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset))

        head_path = head_path_template.format(name=data_name)
        print(f' >>> load classifier head from {head_path} for {data_name}')
        classifier = torch.load(head_path, weights_only=False)
        pretrained.classifier = classifier.to(DEVICE)

        def calculate_logits(data_item):
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(d) for d in data_item['input_ids']], 
                batch_first=True, 
                padding_value=tokenizer.pad_token_id,
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(d) for d in data_item['attention_mask']],  
                batch_first=True, 
                padding_value=0,
            )

            score = pretrained(
                input_ids.to(pretrained.device),
                attention_mask.to(pretrained.device),
            ).logits.cpu().numpy()

            return {
                'predictions': score,
                'label_ids': data_item['label']
            }
    
        dataset = dataset.map(
            lambda x: calculate_logits(x),
            batched=True,
            batch_size=4,
        )
        
        ans = eval.compute_single_metrics(
            utils.SimpleNamespace(
                predictions=torch.tensor(dataset['predictions']),
                label_ids=np.array(dataset['label_ids'])
            ), data_name
        )['averaged_scores']
        metrics[data_name] = 100*float(f"{ans:.4f}")
    
    utils.save_excel(metrics, args.outdir)

@torch.inference_mode()
def run_base2(
    args,
    load_head=True,
): 
    
    for model_name, model_to_merge in zip(args.models_name, args.models_to_merge):
        args.base_model = model_to_merge
        run_pretrained(args)

@torch.inference_mode()
def run_merge(
    args,
):

    args.model_path_template='../roberta/{name}/roberta-base_lr1e-05'
    args.head_path_template='../roberta/{name}/roberta-base_lr1e-05/classifier_head.pt'
    expert_performance_key = 'base'
    if 'large' in args.base_model:
        args.model_path_template='../roberta_large/{name}/roberta-large_lr1e-05'
        args.head_path_template='../roberta_large/{name}/roberta-large_lr1e-05/classifier_head.pt'
        expert_performance_key = 'large'

    if 'base' in args.base_model:
        expert_score = expert_performance['base']
        log_name = None
    elif 'large' in args.base_model:
        expert_score = expert_performance['large']
        log_name = 'large'

    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])
    # \theta_t
    models_finetuned = {
        name: utils.load_classifier(
            args.model_path_template.format(name=name)
        ).to(DEVICE)
        for name in args.models_name
    }
    # \theta_*
    models_to_merge = [
        models_finetuned[name].to(DEVICE)
        for name in args.src_merge
    ]
    base_model = utils.load_classifier(args.base_model).to(DEVICE)

    # 保留原始 base_model 名字符串（之后 args.base_model 会被覆盖为 param 对象）
    base_model_name_str = args.base_model

    # 如果提供了融合模型路径，用它替换预训练模型作为 base_model
    merged_model_path = getattr(args, 'merged_model_path', None)
    if merged_model_path is not None and merged_model_path != "":
        print(f'>>> Using merged model from {merged_model_path} as base_model')
        merged_base = utils.load_classifier(merged_model_path).to(DEVICE)
        if hasattr(base_model, 'classifier'):
            merged_base.classifier = base_model.classifier
        base_model = merged_base
        base_model_name_str = merged_model_path

    args.base_model = param(base_model)
    args.models_to_merge = [param(m) for m in models_to_merge]
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    models_to_merge_dict = {}
    for name, model in zip(args.src_merge, args.models_to_merge):
        models_to_merge_dict[name] = model
    args.models_to_merge_dict = models_to_merge_dict

    if args.merge_method in ('esm_m', 'esm_r'):
        principal_data_path = getattr(args, 'principal_data_path', None) or f'{BASE_DATA_DIR}/data/validation.json'
        principal_direction_dict = get_principal_direction(
            args.models_name, args.base_model.param_dict, args.model_path_template,
            data_path=principal_data_path,
            proxy_num=getattr(args, 'prototype_proxy_num', 32),
        )
        args.principal_direction_dict = principal_direction_dict

    # 3. merge
    merger = MergingMethod(**args)

    # === ESM-R (routing) branch: dynamic routing, no weight fusion ===
    if args.merge_method == 'esm_r':
        prototype_data_path = getattr(args, 'prototype_data_path', None)

        expert_dict, base_param_dict = merger.esm_r(
            base_model=args.base_model,
            models_to_merge_dict=args.models_to_merge_dict,
            principal_direction_dict=principal_direction_dict,
            scaling=1.0,
            orthogonalize_v=True,
            rank=getattr(args, 'rank', 0),
        )

        # Collect prototypes for routing (not needed in oracle/base mode)
        eval_mode = getattr(args, 'mode', 'route')
        proto_proxy_num = getattr(args, 'prototype_proxy_num', 32)
        prototype_dict = None
        if eval_mode == "route":
            proto_data_path = prototype_data_path if prototype_data_path else f'{BASE_DATA_DIR}/data/validation.json'
            prototype_dict = merger.collect_prototypes_for_moe(
                base_model=base_model,
                finetuned_models=models_finetuned,
                val_data_path=proto_data_path,
                proxy_num=proto_proxy_num,
                device=DEVICE,
            )

        if args.data_path is not None:
            metrics = {
                "method": "esm_r",
                "seed": args.seed,
                "prototype_proxy_num": proto_proxy_num,
            }

            data = utils.from_json(args.data_path)
            eval_pred = defaultdict(lambda: defaultdict(list))

            # Load base model once, register MoE hooks once
            eval_model = utils.load_classifier(base_model_name_str).to(DEVICE)
            eval_model.load_state_dict(args.base_model.param_dict, strict=False)
            evaluator = ESMMoEEvaluator(
                eval_model, expert_dict,
                prototype_dict=prototype_dict,
                mode=eval_mode,
            )

            # Group data by task
            task_data = defaultdict(list)
            for data_item in data:
                data_id = data_item['dataset_ids']
                data_name = list(eval.glue_data_id_map.keys())[data_id]
                task_data[data_name].append(data_item)

            batch_size = getattr(args, 'batch_size', 1)

            for data_name, samples in tqdm.tqdm(task_data.items(), desc='infer glue (esm_r)'):
                evaluator.current_task = data_name  # For routing accuracy tracking

                # Load classifier head for this task
                classifier = torch.load(
                    args.head_path_template.format(name=data_name),
                    weights_only=False,
                )
                eval_model.classifier = classifier.to(DEVICE)

                for i in range(0, len(samples), batch_size):
                    batch = samples[i:i + batch_size]

                    # Pad sequences in batch
                    input_ids = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(d['input_ids']) for d in batch],
                        batch_first=True, padding_value=1,
                    ).to(DEVICE)
                    attention_mask = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(d['attention_mask']) for d in batch],
                        batch_first=True, padding_value=0,
                    ).to(DEVICE)

                    if eval_mode == "oracle":
                        evaluator.current_oracle_task = data_name

                    logits = evaluator(input_ids, attention_mask).cpu().numpy()
                    for j, d in enumerate(batch):
                        eval_pred[data_name]['predictions'].append(logits[j:j+1])
                        eval_pred[data_name]['label_ids'].append(d['label'])

            evaluator.clear_hooks()

            # Display and save routing accuracy (skip for oracle since it's always 100%)
            evaluator.print_routing_accuracy()
            if eval_mode != "oracle":
                rank = getattr(args, 'rank', 0)
                seed = getattr(args, 'seed', 0)
                routing_filename = f'routing_accuracy_rank{rank}_seed{seed}.json'
                routing_outpath = os.path.join(args.outdir, routing_filename)
                evaluator.save_routing_accuracy(routing_outpath)

            avg_score = []
            avg_score_abs = []
            for data_name in eval_pred.keys():
                ans = eval.compute_single_metrics(
                    utils.SimpleNamespace(
                        predictions=np.concatenate(eval_pred[data_name]['predictions']),
                        label_ids=np.array(eval_pred[data_name]['label_ids'])
                    ), data_name
                )['averaged_scores']
                abs_score = ans * 100
                normalized_score = abs_score / expert_score[data_name]
                avg_score.append(normalized_score)
                avg_score_abs.append(abs_score)
                metrics[data_name] = 100*float(f"{normalized_score:.4f}")
                metrics[f"{data_name}_abs"] = float(f"{abs_score:.2f}")

            avg_score = sum(avg_score) / len(avg_score)
            metrics['avg'] = 100*float(f"{avg_score:.4f}")
            avg_score_abs = sum(avg_score_abs) / len(avg_score_abs)
            metrics['avg_abs'] = float(f"{avg_score_abs:.2f}")
            utils.save_excel(metrics, args.outdir, log_name)

        return  # ESM-R done (no weight fusion, no scaling search)

    # === Regular merge methods (weight fusion) ===
    merge_method = getattr(merger, args.merge_method)
    merged_param = merge_method(**args)

    if args.merge_method == 'esm_m':

        merged_task_vector = merged_param - args.base_model

        best_scaling, best_score = search_scaling_for_merge(
            base_model=args.base_model,
            merged_task_vector=merged_task_vector,
            models_finetuned=models_finetuned,
            expert_performance_key=expert_performance_key,
            val_data_path=f'{BASE_DATA_DIR}/data/validation.json',
            left=0.0,
            right=5.0,
            tolerance=0.1,
            num_samples_per_task=64,
            scaling_param_names=None
        )

        merged_param = args.base_model + best_scaling * merged_task_vector

    if args.save_path is not None:
        merged_param.assign(base_model)
        base_model.save_pretrained(args.save_path)

    if args.data_path is not None:

        metrics = {
            "method": args.merge_method,
            "seed": args.seed,
            "prototype_proxy_num": getattr(args, 'prototype_proxy_num', None),
        }

        data = utils.from_json(args.data_path)
        eval_pred = defaultdict(lambda: defaultdict(list))
        for data_item in tqdm.tqdm(data, desc='infer glue'):
            data_id = data_item['dataset_ids']
            data_name = list(eval.glue_data_id_map.keys())[data_id]
            
            def calculate_logits(data_item):
                model = models_finetuned[data_name]
                score = torch.func.functional_call(
                    model, 
                    merged_param.param_dict, 
                    args=(
                        torch.tensor(data_item['input_ids']).unsqueeze(0).to(model.device),
                        torch.tensor(data_item['attention_mask']).unsqueeze(0).to(model.device),
                    ),
                ).logits.cpu().numpy()

                return score
        
            eval_pred[data_name]['predictions'].append(calculate_logits(data_item))
            eval_pred[data_name]['label_ids'].append(data_item['label'])

        avg_score = []
        avg_score_abs = []
        for data_name in eval_pred.keys():
            ans = eval.compute_single_metrics(
                utils.SimpleNamespace(
                    predictions=np.concatenate(eval_pred[data_name]['predictions']),
                    label_ids=np.array(eval_pred[data_name]['label_ids'])
                ), data_name
            )['averaged_scores']
            abs_score = ans * 100
            normalized_score = abs_score / expert_score[data_name]
            avg_score.append(normalized_score)
            avg_score_abs.append(abs_score)
            metrics[data_name] = 100*float(f"{normalized_score:.4f}")
            metrics[f"{data_name}_abs"] = float(f"{abs_score:.2f}")

        avg_score = sum(avg_score) / len(avg_score)
        metrics['avg'] = 100*float(f"{avg_score:.4f}")
        avg_score_abs = sum(avg_score_abs) / len(avg_score_abs)
        metrics['avg_abs'] = float(f"{avg_score_abs:.2f}")

        utils.save_excel(metrics, args.outdir, log_name)

def main(
    *, 
    models_to_merge: list[str], 
    models_name: list[str],
    src_merge: list[str],
    yaml_file: str = None,
    exclude_param: list[str] = None, 
    data_path: str = None,
    seed: int=10,
    base_model: str = 'roberta-base',
    scaling: list[float] = None,
    mask_rate: float = None,
    mask_scale: float = None,
    mask_strategy: str = None,
    outdir: str = None,
    save_path: str = None,
    prototype_data_path: str = None,
    prototype_proxy_num: int = 32,
    principal_data_path: str = None,
    merged_model_path: str = None,
    mode: str = "route",
    rank: int = 0,
    batch_size: int = 1,
):

    global args
    keys, _, _, values = inspect.getargvalues(inspect.currentframe())

    utils.fix_seed(seed)

    if models_to_merge[0] == 'NONE':
        args = utils.SimpleNamespace(**{
            k: values.get(k) for k in keys
        })
        run_pretrained(args, load_head=True)
    elif yaml_file is None:
        args = utils.SimpleNamespace(**{
            k: values.get(k) for k in keys
        })
        run_base2(args, load_head=True)
    else:
        merge_config = utils.from_yaml(yaml_file)    
        args = {
            k: values.get(k, merge_config.get(k)) 
            for k in set(keys).union(merge_config)
        }
        args = {
            k: merge_config.get(k, None)
            if args[k] is None else args[k]
            for k in args.keys()
        }
        args = utils.SimpleNamespace(**args)

        print('>>> args\n', args)

        if args.scaling is not None and isinstance(args.scaling, list) and len(args.scaling) == 1:
            args.scaling = args.scaling[0]


        run_merge(args)


if __name__ == '__main__':
    import defopt
    defopt.run(main)