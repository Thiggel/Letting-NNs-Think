from typing import Protocol
from torch import nn, Tensor
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

from experiment.layers.NormalizedGemmaDecoderLayer import CanNormalize
from experiment.layers import NormalizedGemmaDecoderLayer


class NormalizedLanguageModelAdapterProtocol(Protocol):
    model: GemmaForCausalLM

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...

    def normalize(self, tensor: Tensor, dim: int) -> Tensor: ...


class NormalizedLanguageModelAdapter(CanNormalize):
    def _add_normalization(
        self: NormalizedLanguageModelAdapterProtocol, model: nn.Module
    ):
        layers = self.get_decoder_layers(model)

        for idx in range(len(layers)):
            layer = layers[idx]
            new_layer = NormalizedGemmaDecoderLayer(model.config, idx)
            new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
            new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            layers[idx] = new_layer

        model = self.set_decoder_layers(model, layers)

        return model

    def normalize_weights(self: NormalizedLanguageModelAdapterProtocol):
        self.model.get_input_embeddings().weight.data.copy_(
            self.normalize(self.model.get_input_embeddings().weight.data, 1)
        )

        self.model.get_output_embeddings().weight.data.copy_(
            self.normalize(self.model.get_output_embeddings().weight.data, 1)
        )

        for layer in self.get_decoder_layers(self.model):
            layer.self_attn.normalize_weights()
            layer.mlp.normalize_weights()
