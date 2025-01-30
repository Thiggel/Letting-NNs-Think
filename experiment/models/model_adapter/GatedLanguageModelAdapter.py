from typing import Protocol
from torch import nn

from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from experiment.layers.gated_layers import GatedGemmaDecoderLayer, GatedGPTNeoXLayer


class GatedLanguageModelAdapterProtocol(Protocol):
    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...

    def _get_all_recurrent_layers(self, model: nn.Module) -> list[int]: ...


class GatedLanguageModelAdapter:
    def _add_gating(self: GatedLanguageModelAdapterProtocol, model: nn.Module):
        recurrent_layer_indices = self._get_all_recurrent_layers(model)
        layers = self.get_decoder_layers(model)

        for idx in recurrent_layer_indices:
            layer = layers[idx]
            if isinstance(layer, GemmaDecoderLayer):
                new_layer = GatedGemmaDecoderLayer(model.config, idx)
                new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
                new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            elif isinstance(layer, GPTNeoXLayer):
                new_layer = GatedGPTNeoXLayer(model.config, idx)
                new_layer.self_attn.load_state_dict(layer.attention.state_dict())
                new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            else:
                raise ValueError("Unknown layer type")

            layers[idx] = new_layer

        model = self.set_decoder_layers(model, layers)

        return model
