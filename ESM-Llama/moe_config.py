"""
MoE Configuration for MergeBench

Hyperparameters for Mixture of Experts architecture:
- base_model_path: Path to the main/merged model (empty string uses pretrained model)
- moe_rank: Rank for LoRA decomposition (default 64)
- moe_scaling: Scaling coefficient for expert LoRA parameters
- prototype_path: Path to saved prototypes file
- layer_stats_path: Path to layer stats file (from capture_moe_layer_stats.py)
"""

import os

# Default paths
MODEL_BASE = 'meta-llama/Llama-3.2-3B'
MODEL_LOCAL = os.environ.get('BASE_MODEL_PATH', MODEL_BASE)

# Task configurations (same as capture_layer_stats.py)
TASK_CONFIGS = {
    'instruction': {
        'hf_dataset': 'MergeBench/instruction_val',
        'format_keys': {'instruction_key': 'instruction', 'output_key': 'output'},
        'ft_model': 'MergeBench/Llama-3.2-3B_instruction',
    },
    'math': {
        'hf_dataset': 'MergeBench/math_val',
        'format_keys': {'instruction_key': 'query', 'output_key': 'response'},
        'ft_model': 'MergeBench/Llama-3.2-3B_math',
    },
    'coding': {
        'hf_dataset': 'MergeBench/coding_val',
        'format_keys': {'instruction_key': 'instruction', 'output_key': 'response'},
        'ft_model': 'MergeBench/Llama-3.2-3B_coding',
    },
    'safety': {
        'hf_dataset': 'MergeBench/safety_val',
        'format_keys': {'instruction_key': 'prompt', 'output_key': 'response'},
        'ft_model': 'MergeBench/Llama-3.2-3B_safety',
    },
    'multilingual': {
        'hf_dataset': 'MergeBench/multilingual_val',
        'format_keys': {'instruction_key': 'inputs', 'output_key': 'targets'},
        'ft_model': 'MergeBench/Llama-3.2-3B_multilingual',
    },
}

TASK_NAMES = list(TASK_CONFIGS.keys())

LINEAR_LAYER_NAMES = [
    'self_attn.q_proj',
    'self_attn.k_proj',
    'self_attn.v_proj',
    'self_attn.o_proj',
    'mlp.gate_proj',
    'mlp.up_proj',
    'mlp.down_proj',
]


class MoEConfig:
    """Configuration class for MoE architecture."""

    def __init__(
        self,
        base_model_path: str = '',
        moe_rank: int = 64,
        moe_scaling: float = 1.0,
        prototype_path: str = './moe_prototypes_km8.pt',
        layer_stats_path: str = './moe_layer_stats.pt',
        num_experts: int = 5,
        num_samples_capture: int = 64,
        batch_size_capture: int = 16,
        chunk_size_capture: int = 4096,
        routed_layers: list = None,
        routing_mode: str = 'per_layer',
        moe_start_layer: int = 0,
    ):
        self.base_model_path = base_model_path
        self.moe_rank = moe_rank
        self.moe_scaling = moe_scaling
        self.prototype_path = prototype_path
        self.layer_stats_path = layer_stats_path
        self.num_experts = num_experts
        self.num_samples_capture = num_samples_capture
        self.batch_size_capture = batch_size_capture
        self.chunk_size_capture = chunk_size_capture
        self.moe_start_layer = moe_start_layer

        # 路由模式: 'per_layer' (每个线性层独立路由) 或 'per_block' (整个 block 共享路由)
        if routing_mode not in ('per_layer', 'per_block'):
            raise ValueError(f"routing_mode must be 'per_layer' or 'per_block', got {routing_mode}")
        self.routing_mode = routing_mode

        # 指定哪些类型的层使用路由专家，None 表示全部使用
        if routed_layers is None:
            self.routed_layers = list(LINEAR_LAYER_NAMES)
        else:
            for layer in routed_layers:
                assert layer in LINEAR_LAYER_NAMES, \
                    f"Invalid layer name: {layer}. Must be one of {LINEAR_LAYER_NAMES}"
            self.routed_layers = list(routed_layers)

    def get_base_model_path(self):
        """Return actual base model path to use."""
        if self.base_model_path and os.path.exists(self.base_model_path):
            return self.base_model_path
        return MODEL_LOCAL

    def get_routed_layers(self):
        """Return list of layer types that should use routed experts."""
        return self.routed_layers

    def __repr__(self):
        return (f"MoEConfig(base_model_path='{self.base_model_path}', "
                f"actual_path='{self.get_base_model_path()}', "
                f"moe_rank={self.moe_rank}, moe_scaling={self.moe_scaling}, "
                f"prototype_path='{self.prototype_path}', "
                f"layer_stats_path='{self.layer_stats_path}', "
                f"num_experts={self.num_experts}, "
                f"routed_layers={self.routed_layers}, "
                f"routing_mode='{self.routing_mode}', "
                f"moe_start_layer={self.moe_start_layer})")