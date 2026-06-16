import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from collections import defaultdict
import tqdm
import utils
import eval
import numpy as np
import random
from param import param

DEVICE = 'cuda'


def sample_val_data(val_data_path, num_samples_per_task=64, seed=42):
    """
    从验证集中每个任务随机采样固定数量的样本。

    Args:
        val_data_path: str，验证集数据路径
        num_samples_per_task: int，每个任务采样的样本数
        seed: int，随机种子

    Returns:
        sampled_data: list，采样后的数据
    """
    data = utils.from_json(val_data_path)

    # 按任务分组
    task_data = defaultdict(list)
    for item in data:
        task_data[item['dataset_ids']].append(item)

    # 每个任务采样
    random.seed(seed)
    sampled_data = []
    for _task_id, samples in task_data.items():
        if len(samples) <= num_samples_per_task:
            sampled = samples
        else:
            sampled = random.sample(samples, num_samples_per_task)
        sampled_data.extend(sampled)

    random.shuffle(sampled_data)
    return sampled_data


@torch.inference_mode()
def evaluate_on_validation(
    base_model,
    merged_task_vector,
    scaling,
    models_finetuned,
    val_data,
    expert_score,
    scaling_param_names=None,
):
    """
    给定scaling，计算合并模型在验证集上的平均性能。

    Args:
        base_model: param对象，基础模型参数
        merged_task_vector: param对象，合并后的任务向量
        scaling: float，缩放系数
        models_finetuned: dict，每个任务对应的finetuned模型
        val_data: list，验证集数据
        expert_score: dict，每个任务的专家性能（用于归一化）
        scaling_param_names: set，需要搜索scaling的参数名集合。
                            在此集合中的参数使用scaling，其他参数使用1.0

    Returns:
        avg_score: float，所有任务的平均归一化性能
        task_scores: dict，每个任务的原始性能
    """
    # 构建合并参数：指定参数使用scaling，其他参数使用1.0
    if scaling_param_names is not None:
        merged_param_dict = {}
        for param_name in base_model.param_dict.keys():
            base_val = base_model.param_dict[param_name]
            tv_val = merged_task_vector.param_dict[param_name]
            if param_name in scaling_param_names:
                merged_param_dict[param_name] = base_val + scaling * tv_val
            else:
                merged_param_dict[param_name] = base_val + 1.0 * tv_val
        merged_param = param(merged_param_dict)
    else:
        merged_param = base_model + scaling * merged_task_vector

    # 按任务分组数据
    eval_pred = defaultdict(lambda: defaultdict(list))

    for data_item in tqdm.tqdm(val_data, desc=f'eval scaling={scaling:.4f}', leave=False):
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

    # 计算每个任务的性能并取平均
    avg_scores = []
    task_scores = {}

    for data_name in eval_pred.keys():
        result = eval.compute_single_metrics(
            utils.SimpleNamespace(
                predictions=np.concatenate(eval_pred[data_name]['predictions']),
                label_ids=np.array(eval_pred[data_name]['label_ids'])
            ),
            data_name
        )
        score = result['averaged_scores']
        task_scores[data_name] = 100 * float(f"{score:.4f}")

        # 归一化分数
        normalized_score = score * 100 / expert_score[data_name]
        avg_scores.append(normalized_score)

    avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0

    return avg_score, task_scores


@torch.inference_mode()
def binary_search_scaling(
    base_model,
    merged_task_vector,
    models_finetuned,
    val_data_path,
    expert_score,
    left=0.0,
    right=20.0,
    tolerance=0.1,
    num_samples_per_task=64,
    scaling_param_names=None,
):
    """
    使用二分搜索在验证集上寻找最优的scaling系数。

    Args:
        base_model: param对象，基础模型参数
        merged_task_vector: param对象，合并后的任务向量
        models_finetuned: dict，每个任务对应的finetuned模型
        val_data_path: str，验证集数据路径
        expert_score: dict，每个任务的专家性能（用于归一化）
        left: float，搜索区间左端点
        right: float，搜索区间右端点
        tolerance: float，停止条件，当区间长度小于此值时停止
        num_samples_per_task: int，每个任务采样的样本数（None表示使用全部）
        scaling_param_names: set，需要搜索scaling的参数名集合（None表示所有参数都搜索）

    Returns:
        best_scaling: float，最优的scaling系数
        best_score: float，最优性能
        search_history: list，搜索过程中的(scaling, score)记录
    """
    # 加载并采样验证集数据
    if num_samples_per_task is not None:
        val_data = sample_val_data(val_data_path, num_samples_per_task)
    else:
        val_data = utils.from_json(val_data_path)

    search_history = []

    print(f"Starting binary search for scaling in [{left}, {right}]")
    print(f"Tolerance: {tolerance}")
    print(f"Validation data: {len(val_data)} samples")
    if scaling_param_names is not None:
        print(f"Scaling param names count: {len(scaling_param_names)}")
    else:
        print("Scaling all parameters uniformly")

    # 评估端点
    print(f"\nEvaluating left endpoint: scaling={left}")
    left_score, left_task_scores = evaluate_on_validation(
        base_model, merged_task_vector, left,
        models_finetuned, val_data, expert_score,
        scaling_param_names=scaling_param_names,
    )
    search_history.append((left, left_score, left_task_scores))
    print(f"  Average normalized score: {left_score:.4f}")
    print(f"  Task scores: {left_task_scores}")

    print(f"\nEvaluating right endpoint: scaling={right}")
    right_score, right_task_scores = evaluate_on_validation(
        base_model, merged_task_vector, right,
        models_finetuned, val_data, expert_score,
        scaling_param_names=scaling_param_names,
    )
    search_history.append((right, right_score, right_task_scores))
    print(f"  Average normalized score: {right_score:.4f}")
    print(f"  Task scores: {right_task_scores}")

    # 三分搜索
    iteration = 0
    while right - left > tolerance:
        iteration += 1

        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0

        print(f"\n--- Iteration {iteration} ---")
        print(f"Current interval: [{left:.4f}, {right:.4f}], length={right-left:.4f}")

        print(f"Evaluating m1: scaling={m1:.4f}")
        m1_score, m1_task_scores = evaluate_on_validation(
            base_model, merged_task_vector, m1,
            models_finetuned, val_data, expert_score,
            scaling_param_names=scaling_param_names,
        )
        search_history.append((m1, m1_score, m1_task_scores))
        print(f"  Average normalized score: {m1_score:.4f}")

        print(f"Evaluating m2: scaling={m2:.4f}")
        m2_score, m2_task_scores = evaluate_on_validation(
            base_model, merged_task_vector, m2,
            models_finetuned, val_data, expert_score,
            scaling_param_names=scaling_param_names,
        )
        search_history.append((m2, m2_score, m2_task_scores))
        print(f"  Average normalized score: {m2_score:.4f}")

        if m1_score >= m2_score:
            right = m2
            print(f"  -> Keep left part: [{left:.4f}, {right:.4f}]")
        else:
            left = m1
            print(f"  -> Keep right part: [{left:.4f}, {right:.4f}]")

    # 确定最优值
    best_scaling = (left + right) / 2.0
    print(f"\nEvaluating final center point: scaling={best_scaling:.4f}")
    best_score, best_task_scores = evaluate_on_validation(
        base_model, merged_task_vector, best_scaling,
        models_finetuned, val_data, expert_score,
        scaling_param_names=scaling_param_names,
    )
    search_history.append((best_scaling, best_score, best_task_scores))
    print(f"  Average normalized score: {best_score:.4f}")
    print(f"  Task scores: {best_task_scores}")

    # 从搜索历史中找到最优值
    best_entry = max(search_history, key=lambda x: x[1])
    best_scaling, best_score, _ = best_entry

    # 打印搜索总结
    print("\n" + "="*60)
    print("SEARCH SUMMARY")
    print("="*60)
    print(f"{'Scaling':>12} {'Avg Score':>12} {'Task Scores'}")
    print("-"*60)
    seen = set()
    for s, score, task_scores in search_history:
        if s not in seen:
            seen.add(s)
            task_str = ', '.join([f"{k}={v:.2f}" for k, v in sorted(task_scores.items())])
            print(f"{s:>12.4f} {score:>12.4f} {task_str}")
    print("="*60)
    print(f"\nBest scaling: {best_scaling:.4f}")
    print(f"Best average normalized score: {best_score:.4f}")

    return best_scaling, best_score, search_history


@torch.inference_mode()
def search_scaling_for_merge(
    base_model,
    merged_task_vector,
    models_finetuned,
    expert_performance_key='base',
    val_data_path='data/validation.json',
    left=0.0,
    right=20.0,
    tolerance=0.1,
    num_samples_per_task=64,
    scaling_param_names=None,
):
    """
    为模型融合搜索最优scaling系数的入口函数。

    Args:
        base_model: param对象或nn.Module，基础模型
        merged_task_vector: param对象，合并后的任务向量
        models_finetuned: dict，每个任务对应的finetuned模型
        expert_performance_key: str，'base'或'large'，用于选择专家性能数据
        val_data_path: str，验证集数据路径
        left: float，搜索区间左端点
        right: float，搜索区间右端点
        tolerance: float，停止条件
        num_samples_per_task: int，每个任务用于评估的样本数（None表示使用全部）
        scaling_param_names: set，需要搜索scaling的参数名集合（None表示所有参数都搜索）

    Returns:
        best_scaling: float，最优的scaling系数
        best_score: float，最优性能
    """
    # 确保base_model是param对象
    if not isinstance(base_model, param):
        base_model = param(base_model)

    # 确保merged_task_vector是param对象
    if not isinstance(merged_task_vector, param):
        merged_task_vector = param(merged_task_vector)

    # 专家性能数据（用于归一化）
    expert_performance = {
        'base': {
            'cola': 56.52,
            'sst2': 94.72,
            'mrpc': 87.99,
            'stsb': 86.36,
            'qqp': 89.71,
            'mnli': 87.01,
            'qnli': 91.71,
            'rte': 66.43,
        },
        'large': {
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

    expert_score = expert_performance[expert_performance_key]

    best_scaling, best_score, _ = binary_search_scaling(
        base_model=base_model,
        merged_task_vector=merged_task_vector,
        models_finetuned=models_finetuned,
        val_data_path=val_data_path,
        expert_score=expert_score,
        left=left,
        right=right,
        tolerance=tolerance,
        num_samples_per_task=num_samples_per_task,
        scaling_param_names=scaling_param_names,
    )

    return best_scaling, best_score


if __name__ == '__main__':
    # 示例用法
    print("This module provides scaling search functionality.")
    print("Use search_scaling_for_merge() function in your code.")
    print("\nExample:")
    print("  from search_scaling import search_scaling_for_merge")
    print("  best_scaling, best_score = search_scaling_for_merge(")
    print("      base_model=base_model,")
    print("      merged_task_vector=merged_task_vector,")
    print("      models_finetuned=models_finetuned,")
    print("  )")
