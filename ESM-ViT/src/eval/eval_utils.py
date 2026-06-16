import json
import os

import torch
import wandb
from omegaconf import open_dict

from src.eval.eval import evaluate_task_vector, evaluate_task_vector_at_coef, evaluate_task_vector_search
from src.utils.utils import find_optimal_coef
from src.utils.logging import log_results
from src.utils.variables_and_paths import (
    get_finetuned_path,
    get_zeroshot_path,
    get_single_task_accuracies_path,
)


def perform_eval_with_merged_vector(args, task_vector, svd_dict=None, scaling_keys=None):
    assert task_vector is not None, "Task vector should not be None."

    with open_dict(args):
        args.save_dir = os.path.join(args.model_location, args.model)

    ft_accuracies_path = get_single_task_accuracies_path(args.model)
    pretrained_checkpoint = get_zeroshot_path(args.model_location, "MNIST", args.model)

    with open_dict(args):
        with open(ft_accuracies_path) as f:
            args.finetuning_accuracies = json.load(f)
        args.eval_datasets = args.DATASETS_VAL
        args.control_dataset = None

    # evaluate on validation set
    val_metrics = evaluate_task_vector_search(
        task_vector,
        pretrained_checkpoint,
        args,
        svd_dict=svd_dict,
        scaling_keys=scaling_keys
    )

    if args.method.name in ["dummy"]:
        # find scaling factor alpha based on validation accuracy (for Task Arithmetic, TIES, Consensus Merging)
        optimal_coef = find_optimal_coef(
            val_metrics, metric=f"{args.eval_datasets[0]}:top1", minimize=False
        )
        best_val_metrics = val_metrics[optimal_coef]
    else:
        # find scaling factor alpha based on validation accuracy (for Task Arithmetic, TIES, Consensus Merging)
        optimal_coef = find_optimal_coef(
            val_metrics, metric="avg_normalized_top1", minimize=False
        )
        best_val_metrics = val_metrics[optimal_coef]

    print("\n" * 2)

    # Evaluate on the test set with the optimal coefficients / masks
    with open_dict(args):
        args.eval_datasets = args.DATASETS

    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
        svd_dict=svd_dict,
        scaling_keys=scaling_keys
    )

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
    final_results = {
        "test": test_metrics,
        "val": val_metrics,
        "val_best": best_val_metrics,
        "optimal_coef": float(optimal_coef),
    }

    log_results(final_results, args)

    return final_results
