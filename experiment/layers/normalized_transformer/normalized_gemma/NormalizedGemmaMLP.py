import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging

from ..CanNormalize import CanNormalize

logger = logging.get_logger(__name__)


class NormalizedGemmaMLP(nn.Module, CanNormalize):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_activation is None:
            logger.warning_once(
                "`config.hidden_act` is ignored, you should use `config.hidden_activation` instead.\n"
                "Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use\n"
                "`config.hidden_activation` if you want to override this behaviour.\n"
                "See https://github.com/huggingface/transformers/pull/29402 for more details."
            )
            config.hidden_activation = "gelu_pytorch_tanh"
        hidden_activation = config.hidden_activation
        self.act_fn = ACT2FN[hidden_activation]

        self.s_init_value = 1.0
        self.s_init_scaling = 1.0
        self.s_up = torch.nn.Parameter(
            self.s_init_scaling * torch.ones(config.intermediate_size)
        )
        self.s_gate = torch.nn.Parameter(
            self.s_init_scaling * torch.ones(config.intermediate_size)
        )
        self.s_scaling = (self.s_init_value / self.s_init_scaling) * (
            self.config.hidden_size**0.5
        )

    def normalize_weights(self):
        self.up_proj.weight.data.copy_(self.normalize(self.up_proj.weight.data))
        self.gate_proj.weight.data.copy_(self.normalize(self.gate_proj.weight.data))

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        gate = gate * self.s_gate * self.s_scaling
        up = up * self.s_up

        output = self.act_fn(gate) * up

        output = self.down_proj(output)

        return output
