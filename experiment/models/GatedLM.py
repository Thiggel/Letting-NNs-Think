from typing import Protocol
import torch
from torch import nn
from transformers import PreTrainedModel

from experiment.configs.ModelConfig import ModelConfig


class GatedLMProtocol(Protocol):
    config: ModelConfig
    model: PreTrainedModel

    def get_decoder_layers(self, model: PreTrainedModel) -> nn.ModuleList: ...

    def log(self, key: str, value: torch.Tensor): ...


class GatedLM(GatedLMProtocol):
    def get_gate_loss(self):
        layers = self.get_decoder_layers(self.model)

        loss = torch.tensor(0.0)

        if self.config.use_gating:
            for layer in layers:
                if hasattr(layer, "get_gate_loss"):
                    loss += layer.get_gate_loss()

            self.log("gate_loss", loss)

        return loss
