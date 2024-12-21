from typing import Protocol
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaForCausalLM
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiment.configs import ModelConfig
from experiment.layers.normalized_transformer import CanNormalize, NormalizedLMHead
from experiment.layers.normalized_transformer.normalized_gemma import (
    NormalizedGemmaDecoderLayer,
)
from experiment.layers.normalized_transformer.normalized_gpt_neox import (
    NormalizedGPTNeoXLayer,
)
from experiment.layers.normalized_transformer.normalized_llama import (
    NormalizedLlamaDecoderLayer,
)
from experiment.layers.normalized_transformer.normalized_qwen2 import (
    NormalizedQwen2DecoderLayer,
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

    def _get_recurrent_layer_range(self, model: nn.Module) -> list[tuple[int, int]]: ...


class NormalizedLanguageModelAdapter(CanNormalize):
    def _add_normalization(
        self: NormalizedLanguageModelAdapterProtocol, model: nn.Module
    ):
        if hasattr(model, "lm_head"):
            normalized_head = NormalizedLMHead(model.lm_head)
            delattr(model, "lm_head")
            model.add_module("lm_head", normalized_head)
        elif hasattr(model, "embed_out"):
            normalized_head = NormalizedLMHead(model.embed_out)
            delattr(model, "embed_out")
            model.add_module("embed_out", normalized_head)

        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
            # Save the old layer norm weights
            old_layer_norm = model.gpt_neox.final_layer_norm
            weight_data = old_layer_norm.weight.data.clone()
            bias_data = (
                old_layer_norm.bias.data.clone()
                if old_layer_norm.bias is not None
                else None
            )

            # Create and register new layer norm with saved weights
            new_layer_norm = nn.LayerNorm(
                old_layer_norm.normalized_shape,
                elementwise_affine=True,
                bias=bias_data is not None,
            )
            new_layer_norm.weight.data.copy_(weight_data)
            if bias_data is not None:
                new_layer_norm.bias.data.copy_(bias_data)

            # Properly register the new layer norm
            delattr(model.gpt_neox, "final_layer_norm")
            model.gpt_neox.add_module("final_layer_norm", new_layer_norm)

        layers = self.get_decoder_layers(model)

        for idx in range(len(layers)):
            layer = layers[idx]
            layer_is_recurrent = any(
                [
                    rec_start <= idx < rec_end
                    for rec_start, rec_end in self._get_recurrent_layer_range(model)
                ]
            )

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
            elif isinstance(layer, LlamaDecoderLayer):
                new_layer = NormalizedLlamaDecoderLayer(
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
            elif isinstance(layer, Qwen2DecoderLayer):
                new_layer = NormalizedQwen2DecoderLayer(
                    model.config,
                    idx,
                    use_dynamic_rates=layer_is_recurrent
                    and self.config.use_dynamic_eigen_lrs,
                    use_momentum=layer_is_recurrent and self.config.use_momentum,
                )

            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

            print(f"Layer {idx} changed to normalized layer")

            if self.config.pretrained:
                missing_keys, unexpected_keys = new_layer.load_state_dict(
                    layer.state_dict(), strict=False
                )

                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")

            layers[idx] = new_layer

        model = self.set_decoder_layers(model, layers)

        return model

    def normalize_weights(self: NormalizedLanguageModelAdapterProtocol):
        self.model.get_input_embeddings().weight.data.copy_(
            F.normalize(self.model.get_input_embeddings().weight.data, dim=-1)
        )

        self.model.get_output_embeddings().weight.data.copy_(
            F.normalize(self.model.get_output_embeddings().weight.data, dim=-1)
        )

        for layer in self.get_decoder_layers(self.model):
            if isinstance(layer, RecurrentTransformerLayer):
                for recurrent_layer in layer.layer.layers:
                    if hasattr(recurrent_layer, "self_attn"):
                        recurrent_layer.self_attn.normalize_weights()
                    elif hasattr(recurrent_layer, "attention"):
                        recurrent_layer.attention.normalize_weights()

                        recurrent_layer.mlp.normalize_weights()
            else:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.normalize_weights()
                elif hasattr(layer, "attention"):
                    layer.attention.normalize_weights()

                layer.mlp.normalize_weights()
