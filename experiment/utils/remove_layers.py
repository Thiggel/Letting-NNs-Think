from transformers import AutoModelForCausalLM
from torch import nn


def remove_layers(model: AutoModelForCausalLM, remove_layers: list[int]):
    for i in remove_layers:
        model.transformer.h[i] = nn.Identity()
