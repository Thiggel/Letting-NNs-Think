from typing import Protocol
from torch import nn

from experiment.layers import GatedGemmaDecoderLayer


class GatedLanguageModelAdapterProtocol(Protocol):
    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...

    def _get_recurrent_layer_range(self, model: nn.Module) -> tuple[int, int]: ...


class GatedLanguageModelAdapter:
    def _add_gating(self: GatedLanguageModelAdapterProtocol, model: nn.Module):
        start, end = self._get_recurrent_layer_range(model)
        layers = self.get_decoder_layers(model)

        for idx in range(start, end):
            layer = layers[idx]
            new_layer = GatedGemmaDecoderLayer(model.config, idx)
            new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
            new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            layers[idx] = new_layer

        model = self.set_decoder_layers(model, layers)

        return model
