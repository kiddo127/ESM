"""ESM (Essential Subspace Merging) — merge algorithm.

Concatenates per-task ESD factors, applies inter-task and inter-column scaling,
then orthogonalizes via polar decomposition (SVD) before reconstruction.
"""

import torch


def esm_merge(task_vector_dict, config, cov_dict, principal_direction_dict):
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    norm_power = 2
    print("Computing Essential Subspace Merging...")
    with torch.no_grad():
        new_vector = {}
        principal_direction_keys = list(principal_direction_dict.values())[0].keys()
        for key in list(task_vector_dict.values())[0].vector:
            print(f"Computing ESM for {key}...")
            sum_u = []
            sum_s = []
            sum_v = []
            norms_v = []
            new_vector[key] = {}
            for i, (dataset, task_vector) in enumerate(task_vector_dict.items()):
                vec = task_vector.vector[key].to(device)
                if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                    if key in principal_direction_keys:
                        if key.endswith(".proj"):
                            reduced_index_s = int(vec.shape[1] * sv_reduction)
                        else:
                            reduced_index_s = int(vec.shape[0] * sv_reduction)
                        principal_directions = principal_direction_dict[dataset][key]['eigenvectors'].cuda()
                        u = principal_directions
                        if key.endswith(".proj"):
                            v = principal_directions.T @ vec.T
                        else:
                            v = principal_directions.T @ vec

                        # inter-task scaling
                        norm_v = torch.norm(v)
                        norms_v.append(norm_v)
                        v = v * pow(norm_v, norm_power)

                        eigenvalues = principal_direction_dict[dataset][key]['eigenvalues'][:reduced_index_s].cuda()
                        eigenvalues_scale = torch.pow(eigenvalues / eigenvalues.mean(), 0.15)
                        eigenvalues_scale = eigenvalues_scale / eigenvalues_scale.mean()
                        sum_u.append(u[:, :reduced_index_s] * eigenvalues_scale.unsqueeze(0))
                        sum_v.append(v[:reduced_index_s, :] * eigenvalues_scale.unsqueeze(1))
                    else:
                        u, s, v = torch.linalg.svd(vec, full_matrices=False)
                        reduced_index_s = int(s.shape[0] * sv_reduction)
                        sum_u.append(u[:, :reduced_index_s])
                        sum_s.append(s[:reduced_index_s])
                        sum_v.append(v[:reduced_index_s, :])

                else:  # conv/norm/bias
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                if key in principal_direction_keys:
                    sum_u = torch.cat(sum_u, dim=1)
                    sum_v = torch.cat(sum_v, dim=0)

                    # inter-task scaling
                    norms_v_mean = torch.tensor(norms_v).mean()
                    sum_v = sum_v / pow(norms_v_mean, norm_power)

                    # inter-column scaling (task consensus)
                    col_norms = torch.norm(sum_v, dim=0)
                    col_scales = (col_norms / col_norms.mean()).unsqueeze(0)
                    sum_v = sum_v * pow(col_scales, norm_power)

                    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                    U_ortho = u_u @ v_u * s_u.mean()
                    V_ortho = u_v @ v_v * s_v.mean()

                    new_vector[key] = (U_ortho @ V_ortho * 2).cuda()

                    if key.endswith(".proj"):
                        new_vector[key] = new_vector[key].T
                else:
                    sum_u = torch.cat(sum_u, dim=1)
                    sum_s = torch.cat(sum_s, dim=0)
                    sum_v = torch.cat(sum_v, dim=0)
                    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                    new_vector[key] = torch.linalg.multi_dot(
                        (
                            u_u,
                            v_u,
                            torch.diag(sum_s),
                            u_v,
                            v_v,
                        )
                    )

    # inter-layer scaling (task consensus)
    norms = {
        'attn.in_proj_weight': [],
        'attn.out_proj.weight': [],
        'mlp.c_fc.weight': [],
        'mlp.c_proj.weight': [],
    }
    for key in new_vector.keys():
        for norms_key in norms.keys():
            if norms_key in key:
                norms[norms_key].append(torch.norm(new_vector[key]))
    for key in norms.keys():
        norms[key] = torch.tensor(norms[key]).cuda().mean()
    for key in new_vector.keys():
        for norms_key in norms.keys():
            if norms_key in key:
                scale = torch.norm(new_vector[key]) / norms[norms_key]
                new_vector[key] *= pow(scale, norm_power)

    return new_vector
