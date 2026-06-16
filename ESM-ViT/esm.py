import os
import torch
import numpy as np
from pprint import pprint

import hydra

import wandb
wandb.init(mode="offline")
from omegaconf import DictConfig, OmegaConf

from src.eval.aggregation import create_task_vector_dict, create_task_vector_principal_direction
from src.eval.eval_utils import perform_eval_with_merged_vector
from src.utils.variables_and_paths import ALL_DATASETS, get_zeroshot_path

from essential_subspace_decomposition import esd


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def esm(cfg: DictConfig) -> None:

    if cfg.DATASETS == "":
        cfg.DATASETS = ALL_DATASETS[: cfg.num_tasks]
    else:
        cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    OmegaConf.set_struct(cfg, True)

    # set up experiment for WandB
    print(cfg.method.full_name)
    print()
    wandb.init(
        config=OmegaConf.to_container(cfg),
        mode=cfg.wandb.mode,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        dir="logs/",
    )
    wandb.config.update({"method.full_name1": cfg.method.full_name})
    wandb.config.update({"method.keep": cfg.method.k})
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "save_dir", os.path.join(cfg.model_location, cfg.model), force_add=True)


    # create initial task vector
    print("*" * 100)
    print("*" * 37, "Creating initial task vector dict", "*" * 37)
    print("*" * 100)
    print("\n" * 3)
    task_vector_dict = create_task_vector_dict(cfg)


    # ESD: Essential Subspace Decomposition
    print("*" * 100)
    print("*" * 34, "Essential Subspace Decomposition (ESD)", "*" * 33)
    print("*" * 100)
    esd_dict = esd(cfg, task_vector_dict)
    print(list(esd_dict.values())[0].keys())

    # ratios = np.linspace(0, 0.5, 100)
    # inform_ratios = {dataset: [[] for i in range(len(ratios))] for dataset in cfg.DATASETS_VAL}
    # for dataset in cfg.DATASETS_VAL:
    #     for _, modlue_dict in principal_direction_dict[dataset].items():
    #         eigenvalues = modlue_dict['eigenvalues']
    #         for idx, ratio in enumerate(ratios):
    #             inform_ratios[dataset][idx].append(eigenvalues[:int(eigenvalues.shape[0] * ratio)].sum() / eigenvalues.sum())
    # keep_ratios = torch.zeros((len(ratios), len(cfg.DATASETS_VAL)))
    # for d_idx, dataset in enumerate(cfg.DATASETS_VAL):
    #     for idx, ratio in enumerate(ratios):
    #         keep_ratios[idx,d_idx] = torch.tensor(inform_ratios[dataset][idx]).mean()
    # print(keep_ratios)
    # np.save(f'results/esd_{cfg.model}.npy', keep_ratios.numpy())
    # exit()


    # ESM merge
    print("*" * 100)
    print("*" * 40, "ESM: Essential Subspace Merging", "*" * 40)
    print("*" * 100)
    print("\n" * 3)
    merge_vector_dict, svd_dict = create_task_vector_principal_direction(cfg, task_vector_dict, None, esd_dict)


    # perform evaluation and log results
    print("*" * 100)
    print("*" * 39, "Starting Evaluation", "*" * 39)
    print("*" * 100)
    additive_accuracies = perform_eval_with_merged_vector(
        cfg, merge_vector_dict, svd_dict, list(esd_dict.values())[0].keys()
    )
    pprint(additive_accuracies, width=1)
    wandb.log(additive_accuracies)

    # save fused main model using the optimal coefficient from validation
    if cfg.save_main_model:
        print("*" * 100)
        print("*" * 37, "Saving fused main model (optimal coef)", "*" * 37)
        print("*" * 100)
        optimal_coef = additive_accuracies.get("optimal_coef", 1.0)
        pretrained_checkpoint = get_zeroshot_path(cfg.model_location, "MNIST", cfg.model)
        fused_model = merge_vector_dict.apply_to(
            pretrained_checkpoint, scaling_coef=optimal_coef, args=cfg
        )
        save_path = cfg.main_model_path if cfg.main_model_path else \
            os.path.join(cfg.model_location, cfg.model, f"fused_model_{cfg.num_tasks}tasks.pt")
        fused_model.save(save_path)
        print(f"Fused model saved to {save_path} (optimal_coef={optimal_coef:.4f})")

    wandb.finish(quiet=True)


if __name__ == "__main__":
    esm()
