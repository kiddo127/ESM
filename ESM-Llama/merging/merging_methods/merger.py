import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class Merger(nn.Module):
    def __init__(self, base_model, ft_models, save_path):
        super().__init__()

        self.base_model_name = base_model
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, torch_dtype="bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.ft_ckpts = [AutoModelForCausalLM.from_pretrained(ft_model, torch_dtype="bfloat16") for ft_model in ft_models]
        self.save_path = save_path

    def merge(self, **kwargs):
        pass
