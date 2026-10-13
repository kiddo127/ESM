import torch

def flatten_ckpt_into_vec(ckpt):
    vec = []
    for param in ckpt.values():
        vec.append(param.flatten())
    return torch.cat(vec)

def select_trainable_params(model):
    params = {}

    for n, p in model.named_parameters():
        if 'embed' not in n and 'Embedding' not in n:
            params[n] = p
                    
    return params

def get_task_vector(ft_model, base_model):
    ft_model.to('cpu')
    base_model.to('cpu')

    ft_params = select_trainable_params(ft_model)
    base_params = select_trainable_params(base_model)

    ft_vec = flatten_ckpt_into_vec(ft_params)
    base_vec = flatten_ckpt_into_vec(base_params)

    return ft_vec - base_vec

def vector_to_state_dict(vec, pretrained_model, return_dict=False):
    i = 0
    vec.to('cpu')
    pretrained_model.to('cpu')
    for k, v in pretrained_model.state_dict().items():
        if 'embed' not in k.lower() and 'lm_head' not in k:
            if torch.nonzero(v).size(0) == 0:
                continue
            vec[i:i+v.numel()].reshape(v.shape).to(pretrained_model.device)
            pretrained_model.state_dict()[k] += vec[i:i+v.numel()].reshape(v.shape)
            i += v.numel()

    if return_dict:
        return pretrained_model.state_dict()
    else:
        return pretrained_model
