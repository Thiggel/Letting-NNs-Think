from typing import Protocol, Optional, Annotated
import torch
from torch import nn

from experiment.layers import (
    DynamicVeraLayer,
    SequentialTransformerLayer,
)
from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig, LayerRange


class RecurrentLanguageModelAdapterProtocol(Protocol):
    config: ModelConfig
    model: nn.Module
    device: torch.device
    recurrent_layer_idx: int

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...

    def _get_recurrent_layer_range(self, model: nn.Module) -> list[tuple[int, int]]: ...

    def _add_recurrence_to_layer(
        self, model: nn.Module, start: int, end: int
    ) -> nn.Module: ...

    def _create_mamba_layer(self, num_layers: int) -> SequentialTransformerLayer: ...

    def _get_layer_range(
        self, model: nn.Module, layer_range: Optional[Annotated[str, LayerRange]]
    ) -> list[tuple[int, int]]: ...

    def _get_all_layers(
        self, model: nn.Module, layer_range: Optional[Annotated[str, LayerRange]]
    ) -> list[int]: ...


class RecurrentLanguageModelAdapter:
    def _add_recurrence(self: RecurrentLanguageModelAdapterProtocol, model: nn.Module):
        """Add recurrent layers to the model"""
        recurrent_layer_ranges = self._get_recurrent_layer_range(model)

        for start, end in recurrent_layer_ranges:
            model = self._add_recurrence_to_layer(model, start, end)

        return model

    def _add_recurrence_to_layer(
        self: RecurrentLanguageModelAdapterProtocol,
        model: nn.Module,
        start: int,
        end: int,
    ):
        """Add recurrent layers to the model"""
        layers = self.get_decoder_layers(model)
        recurrent_layers = layers[start:end]

        if self.config.recurrent_mode == "mamba":
            recurrent_layer: nn.Module = self._create_mamba_layer(len(recurrent_layers))
        else:
            recurrent_layer = SequentialTransformerLayer(*recurrent_layers)

        if self.config.use_dynamic_vera:
            recurrent_layer = DynamicVeraLayer(
                recurrent_layer,
                model.config.hidden_size,
                self.config.vera_r,
                self.device,
            )

        layers[start] = RecurrentTransformerLayer(
            recurrent_layer,
            config=self.config,
            hidden_size=model.config.hidden_size,
        )

        # Remove the original layers that were made recurrent
        for i in range(start + 1, end):
            print(i)
            layers.pop(start + 1)

        model = self.set_decoder_layers(model, layers)

        self.recurrent_layer_idx = start

        return model

    def _get_recurrent_layer_range(
        self: RecurrentLanguageModelAdapterProtocol, model: nn.Module
    ) -> list[tuple[int, int]]:
        return self._get_layer_range(model, self.config.make_layers_recurrent)

    def _get_all_recurrent_layers(
        self: RecurrentLanguageModelAdapterProtocol, model: nn.Module
    ) -> list[int]:
        return self._get_all_layers(model, self.config.make_layers_recurrent)
