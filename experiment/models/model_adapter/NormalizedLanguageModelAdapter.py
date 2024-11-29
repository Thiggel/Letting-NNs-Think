from typing import Protocol
from torch import nn, Tensor
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gpt_neox import GPTNeoXLayer

from experiment.configs import ModelConfig
from experiment.layers.normalized_transformer import CanNormalize, NormalizedLMHead
from experiment.layers.normalized_transformer.normalized_gemma import (
    NormalizedGemmaDecoderLayer,
)
from experiment.layers.normalized_transformer.normalized_gpt_neox import (
    NormalizedGPTNeoXLayer,
)
from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer


class NormalizedLanguageModelAdapterProtocol(Protocol):
    model: GemmaForCausalLM
    config: ModelConfig

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...

    def normalize(self, tensor: Tensor, dim: int = -1) -> Tensor: ...

    def _get_recurrent_layer_range(self, model: nn.Module) -> tuple[int, int]: ...


class NormalizedLanguageModelAdapter(CanNormalize):
    def _add_normalization(
        self: NormalizedLanguageModelAdapterProtocol, model: nn.Module
    ):
        model.lm_head = NormalizedLMHead(model.lm_head)

        layers = self.get_decoder_layers(model)

        recurrent_layer_start, recurrent_layer_end = self._get_recurrent_layer_range(
            model
        )

        for idx in range(len(layers)):
            layer = layers[idx]
            layer_is_recurrent = recurrent_layer_start <= idx < recurrent_layer_end
            if isinstance(layer, GemmaDecoderLayer) or isinstance(
                layer, Gemma2DecoderLayer
            ):
                new_layer = NormalizedGemmaDecoderLayer(
                    model.config,
                    idx,
                    use_dynamic_rates=layer_is_recurrent
                    and self.config.use_dynamic_eigen_lrs,
                    use_momentum=layer_is_recurrent and self.config.use_momentum,
                )
            elif isinstance(layer, GPTNeoXLayer):
                new_layer = NormalizedGPTNeoXLayer(
                    model.config,
                    idx,
                    use_dynamic_rates=layer_is_recurrent
                    and self.config.use_dynamic_eigen_lrs,
                    use_momentum=layer_is_recurrent and self.config.use_momentum,
                )

            missing_keys, unexpected_keys = new_layer.load_state_dict(
                layer.state_dict(), strict=False
            )

            print(f"Layer {idx} changed to normalized layer")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

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
                    recurrent_layer.self_attn.normalize_weights()
                    recurrent_layer.mlp.normalize_weights()
            else:
                layer.self_attn.normalize_weights()
                layer.mlp.normalize_weights()
