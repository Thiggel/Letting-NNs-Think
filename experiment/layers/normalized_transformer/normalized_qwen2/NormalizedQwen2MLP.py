import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NormalizedQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.s_init_value = 1.0
        self.s_init_scaling = 1.0
        self.s_up = torch.nn.Parameter(
            self.s_init_scaling * torch.ones(config.intermediate_size)
        )
        self.s_gate = torch.nn.Parameter(
            self.s_init_scaling * torch.ones(config.intermediate_size)
        )
        self.s_scaling = (self.s_init_value / self.s_init_scaling) * (
            config.hidden_size**0.5
        )

    def normalize_weights(self):
        self.up_proj.weight.data.copy_(F.normalize(self.up_proj.weight.data, dim=-1))
        self.gate_proj.weight.data.copy_(
            F.normalize(self.gate_proj.weight.data, dim=-1)
        )
        self.down_proj.weight.data.copy_(F.normalize(self.down_proj.weight.data, dim=0))

    def forward(self, hidden_state):
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)

        gate = gate * self.s_gate * self.s_scaling
        up = up * self.s_up

        return self.down_proj(self.act_fn(gate) * up)
