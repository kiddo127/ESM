import argparse

def create_parser():
    _parser = argparse.ArgumentParser(description='Configuration for MergeBench')

    # DDP
    _parser.add_argument("--local_rank", type=int, default=0)

    # ESM
    _parser.add_argument('--scaling-coef', default=0.9, type=float)
    _parser.add_argument('--layer-stats-path', type=str, default='./layer_stats.pt',
                        help='Path to the layer_stats.pt file from capture_layer_stats.py')
    _parser.add_argument('--sv-reduction', type=float, default=0.1,
                        help='fraction of principal components kept per task (default 0.1)')
    _parser.add_argument('--eig-exp', type=float, default=0.15,
                        help='exponent for the eigenvalue weighting applied to U (default 0.15)')
    _parser.add_argument('--use-eig-weight',
                        type=lambda x: str(x).lower() not in ('0', 'false', 'no'),
                        default=True,
                        help='whether to apply the eigenvalue weighting to U (default True). '
                             'Set False (--use-eig-weight False) to ablate it.')

    _parser.add_argument('--task_names', type=str)
    _parser.add_argument("--save_group", type=str, default=None)

    return _parser

def prepare_args(params):
    kwargs = {}
    if params.algo == 'ESM':
        kwargs['scaling_coef'] = params.scaling_coef
        kwargs['layer_stats_path'] = params.layer_stats_path
        kwargs['sv_reduction'] = params.sv_reduction
        kwargs['eig_exp'] = params.eig_exp
        kwargs['use_eig_weight'] = params.use_eig_weight
        if params.sv_reduction is not None:
            kwargs['sv_reduction'] = params.sv_reduction
    else:
        raise ValueError('No support merging method {}'.format(params.algo))

    return kwargs
