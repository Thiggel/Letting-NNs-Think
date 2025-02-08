from torch import nn
import torch
from experiment.configs.ModelConfig import ModelConfig


class GateLayer(nn.Module):
    """Gate layer that applies gating to module outputs"""

    def __init__(self, d_model: int, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)
        nn.init.normal_(self.gate.weight, std=config.gate_init_std)
        nn.init.constant_(self.gate.bias, config.gate_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x))
