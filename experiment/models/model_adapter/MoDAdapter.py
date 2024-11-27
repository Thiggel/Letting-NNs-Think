from typing import Protocol
from torch import nn

from experiment.layers.mixture_of_depths import MoDLayer
from experiment.configs import ModelConfig


class MoDAdapterProtocol(Protocol):
    config: ModelConfig

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...


class MoDAdapter:
    def _add_mod(self: MoDAdapterProtocol, model: nn.Module):
        model = self.set_decoder_layers(
            model,
            nn.ModuleList(
                [
                    MoDLayer(
                        layer,
                        model,
                        self.config.mod_capacity,
                        self.config.mod_router_hidden_dim,
                        self.config.mod_z_loss_weight,
                        self.config.mod_capacity_loss_weight,
                        reset_mod_loss=(i == 0),
                    )
                    for i, layer in enumerate(self.get_decoder_layers(model))
                ]
            ),
        )

        return model
