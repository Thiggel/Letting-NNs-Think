import torch
from torch import nn
from transformers.activations import ACT2FN

from ..CanNormalize import CanNormalize


class NormalizedGPTNeoXMLP(nn.Module, CanNormalize):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

        self.s_init_value = 1.0
        self.s_init_scaling = 1.0
        self.s_up = torch.nn.Parameter(
            self.s_init_scaling * torch.ones(config.hidden_size * 4)
        )
        self.s_down = torch.nn.Parameter(
            self.s_init_scaling * torch.ones(config.hidden_size)
        )

    def normalize_weights(self):
        self.dense_h_to_4h.weight.data.copy_(
            self.normalize(self.dense_h_to_4h.weight.data)
        )
        self.dense_4h_to_h.weight.data.copy_(
            self.normalize(self.dense_4h_to_h.weight.data, 0)
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states) * self.s_up
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states) * self.s_down
        return hidden_states
