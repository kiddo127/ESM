import os
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig

from src.models.task_vectors import ImageEncoder, NonLinearTaskVector
from src.utils.utils import check_parameterNamesMatch
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path
from src.utils.esm_utils import esm_merge


def get_all_checkpoints(
    config: DictConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    model_dir = config.model_location
    print("Loading checkpoints")
    print("datasets:", config.DATASETS_VAL)
    print("model:", config.model)
    for dataset in config.DATASETS_VAL:
        path = get_finetuned_path(model_dir, dataset, model=config.model)
        if os.path.exists(path):
            print(f"{path} exists")
        else:
            print(f"{path} does not exist")

    params = {
        dataset: torch.load(
            get_finetuned_path(model_dir, dataset, model=config.model),
            map_location="cpu",
        )
        for dataset in config.DATASETS_VAL
    }

    try:
        ptm_check = torch.load(
            get_zeroshot_path(model_dir, "MNISTVal", model=config.model),
            map_location="cpu",
        )
    except:
        ptm_check = ImageEncoder(config.model).state_dict()
        torch.save(
            ptm_check, get_zeroshot_path(model_dir, "MNISTVal", model=config.model)
        )

    return params, ptm_check


def create_task_vector_dict(
    config: DictConfig,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    ft_checks, ptm_check = get_all_checkpoints(config)
    check_parameterNamesMatch(list(ft_checks.values()) + [ptm_check])

    task_vector_dict = {
        dt_name: NonLinearTaskVector(config.model, ptm_check, check)
        for (dt_name, check) in ft_checks.items()
    }

    return task_vector_dict


def create_task_vector_principal_direction(
    config: DictConfig, task_vector_dict=None, cov_dict=None, principal_direction_dict=None
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    print(f"MODEL: {config.model}, METHOD {config.method.name}")
    print(f"=== Essential Subspace Merging ===")

    new_merged_tv = esm_merge(
        task_vector_dict, config, cov_dict, principal_direction_dict
    )

    task_vector = NonLinearTaskVector(model_name=config.model, vector=new_merged_tv)
    print("Norm of task vector: ", task_vector.norm())

    return task_vector, None
