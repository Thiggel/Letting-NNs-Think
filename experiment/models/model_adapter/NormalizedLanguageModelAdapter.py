from typing import Protocol
from torch import nn, Tensor
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

from experiment.configs import ModelConfig
from experiment.layers.NormalizedGemmaDecoderLayer import (
    CanNormalize,
    NormalizedGemmaLMHead,
)
from experiment.layers import NormalizedGemmaDecoderLayer
from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer


class NormalizedLanguageModelAdapterProtocol(Protocol):
    model: GemmaForCausalLM
    config: ModelConfig

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...

    def normalize(self, tensor: Tensor, dim: int) -> Tensor: ...

    def _get_recurrent_layer_range(self, model: nn.Module) -> tuple[int, int]: ...


class NormalizedLanguageModelAdapter(CanNormalize):
    def _add_normalization(
        self: NormalizedLanguageModelAdapterProtocol, model: nn.Module
    ):
        model.lm_head = NormalizedGemmaLMHead(model.lm_head)

        layers = self.get_decoder_layers(model)

        recurrent_layer_start, recurrent_layer_end = self._get_recurrent_layer_range(
            model
        )

        for idx in range(len(layers)):
            layer = layers[idx]
            layer_is_recurrent = recurrent_layer_start <= idx < recurrent_layer_end
            new_layer = NormalizedGemmaDecoderLayer(
                model.config,
                idx,
                use_dynamic_rates=layer_is_recurrent
                and self.config.use_dynamic_eigen_lrs,
                use_momentum=layer_is_recurrent and self.config.use_momentum,
            )
            new_layer.self_attn.load_state_dict(
                layer.self_attn.state_dict(), strict=False
            )
            new_layer.mlp.load_state_dict(layer.mlp.state_dict(), strict=False)
            layers[idx] = new_layer

        model = self.set_decoder_layers(model, layers)

        return model

    def normalize_weights(self: NormalizedLanguageModelAdapterProtocol):
        self.model.get_input_embeddings().weight.data.copy_(
            self.normalize(self.model.get_input_embeddings().weight.data)
        )

        self.model.get_output_embeddings().weight.data.copy_(
            self.normalize(self.model.get_output_embeddings().weight.data)
        )

        for layer in self.get_decoder_layers(self.model):
            if isinstance(layer, RecurrentTransformerLayer):
                for recurrent_layer in layer.layer.layers:
                    print("Normalizing weights for recurrent layer")
                    recurrent_layer.self_attn.normalize_weights()
                    recurrent_layer.mlp.normalize_weights()
            else:
                print("Normalizing weights for layer")
                layer.self_attn.normalize_weights()
                layer.mlp.normalize_weights()
