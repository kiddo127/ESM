import torch
import tqdm
from collections import defaultdict
import utils
import eval
import random


def get_principal_direction_svd(act_data):
    original_shape = act_data.shape
    feature_dim = original_shape[-1]
    if act_data.dim() > 2:
        act_data = act_data.reshape(-1, feature_dim)
    n_samples = act_data.shape[0]
    feature_dim = act_data.shape[1]
    try:
        U, S, Vh = torch.linalg.svd(act_data, full_matrices=False)
        eigenvalues = S**2 / (n_samples - 1)  # λ_i = σ_i²/(n-1)
    except:
        U, S, Vh = torch.linalg.svd(act_data * 0.1, full_matrices=False)
        eigenvalues = S**2 / (n_samples - 1) * 100  # λ_i = σ_i²/(n-1)
    eigenvectors = Vh.T
    if eigenvectors.shape[1] < feature_dim:
        full_eig = torch.eye(feature_dim, device=act_data.device, dtype=act_data.dtype)
        full_eig[:, :eigenvectors.shape[1]] = eigenvectors
        eigenvectors = full_eig
        full_eigenvalues = torch.zeros(feature_dim, device=eigenvalues.device, dtype=eigenvalues.dtype)
        full_eigenvalues[:len(eigenvalues)] = eigenvalues
        eigenvalues = full_eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    return eigenvalues, eigenvectors

def get_activation_hook(layer_name, delta_W, activation_diff_dict=None):
    def hook(module, input_tensor, output_tensor):
        name = layer_name + '.weight'
        x = input_tensor[0]
        original_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, original_shape[-1])

        activation_diff = torch.matmul(x, delta_W.T.to(x.device)).detach()
        if name in activation_diff_dict:
            activation_diff_dict[name] = torch.cat([
                activation_diff_dict[name],
                activation_diff
            ], dim=0)
        else:
            activation_diff_dict[name] = activation_diff

    return hook

def register_hook(base_model_dict, finetuned_model, target_layer_names, activation_diff_dict=None):
    hook_handles = []
    cnt = 0
    finetuned_param_dict = finetuned_model.state_dict()
    for name, module in finetuned_model.named_modules():
        if name in target_layer_names:
            cnt += 1
            weight_name = name + ".weight"
            delta_W = (finetuned_param_dict[weight_name] - base_model_dict[weight_name]).detach()
            handle = module.register_forward_hook(
                get_activation_hook(name, delta_W, activation_diff_dict)
            )
            hook_handles.append(handle)
    assert cnt == len(target_layer_names)
    return hook_handles


@torch.inference_mode()
def get_principal_direction(
    models_name,
    base_model_dict,
    model_path_template,
    data_path="data/validation.json",
    proxy_num=64,
):

    principal_direction_dict = {}
    for name in models_name:
        principal_direction_dict[name] = {}
    activation_diff_dict = {}
    for name in models_name:
        activation_diff_dict[name] = {}

    models_finetuned = {
        name: utils.load_classifier(
            model_path_template.format(name=name)
        ).to('cuda:0')
        for name in models_name
    }

    target_layer_names = []
    for name, module in models_finetuned[models_name[0]].named_modules():
        if (name.endswith(".attention.self.query") or
            name.endswith(".attention.self.key") or
            name.endswith(".attention.self.value") or
            name.endswith(".attention.output.dense") or
            name.endswith(".intermediate.dense") or
            name.endswith(".output.dense")):
            target_layer_names.append(name)
    print(f"Found {len(target_layer_names)} target layers to analyze.")

    data = utils.from_json(data_path)

    task_data = defaultdict(list)
    for data_item in data:
        task_id = data_item['dataset_ids']
        task_data[task_id].append(data_item)

    proxy_data = {}
    for task_id, samples in task_data.items():
        if len(samples) <= proxy_num:
            proxy_data[task_id] = samples
        else:
            proxy_data[task_id] = random.sample(samples, proxy_num)

    hook_handles = {}
    for data_name in list(eval.glue_data_id_map.keys()):
        model = models_finetuned[data_name]
        hook_handles[data_name] = register_hook(
            base_model_dict, model, target_layer_names,
            activation_diff_dict[data_name],
        )

    eval_pred = defaultdict(lambda: defaultdict(list))
    for data_id in proxy_data.keys():
        data_name = list(eval.glue_data_id_map.keys())[data_id]

        def calculate_logits(data_item):
            model = models_finetuned[data_name]
            score = torch.func.functional_call(
                model,
                model.state_dict(),
                args=(
                    torch.tensor(data_item['input_ids']).unsqueeze(0).to(model.device),
                    torch.tensor(data_item['attention_mask']).unsqueeze(0).to(model.device),
                ),
            ).logits.cpu().numpy()
            return score

        for data_item in tqdm.tqdm(proxy_data[data_id], desc=f'get principal direction of task {data_name}'):
            eval_pred[data_name]['predictions'].append(calculate_logits(data_item))
            eval_pred[data_name]['label_ids'].append(data_item['label'])

        for layer_name, output_shift in activation_diff_dict[data_name].items():
            eigenvalues, eigenvectors = get_principal_direction_svd(output_shift)
            principal_direction_dict[data_name][layer_name] = {
                'eigenvalues': eigenvalues.cpu(),
                'eigenvectors': eigenvectors.cpu()
            }

        for handle in hook_handles[data_name]:
            handle.remove()
        hook_handles[data_name].clear()
        activation_diff_dict[data_name] = {}

    return principal_direction_dict

