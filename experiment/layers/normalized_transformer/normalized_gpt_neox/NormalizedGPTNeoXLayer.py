from typing import Optional, Tuple
import torch
from torch import nn
from transformers.cache_utils import Cache

from experiment.layers.normalized_transformer import (
    CanNormalize,
    NormalizedDecoderLayer,
)

from .NormalizedGPTNeoXSdpaAttention import NormalizedGPTNeoXSdpaAttention
from .NormalizedGPTNeoXMLP import NormalizedGPTNeoXMLP


class NormalizedGPTNeoXLayer(nn.Module, CanNormalize, NormalizedDecoderLayer):
    def __init__(
        self,
        config,
        layer_idx,
        use_dynamic_rates: bool = False,
        use_momentum: bool = False,
    ):
        super().__init__()
        self.use_dynamic_rates = use_dynamic_rates
        self.use_momentum = use_momentum

        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = NormalizedGPTNeoXSdpaAttention(config, layer_idx)
        self.mlp = NormalizedGPTNeoXMLP(config)
        self.hidden_size = config.hidden_size

        # Static eigen learning rates (used when not dynamic)
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / (config.hidden_size**0.5)
        self.attn_alpha = torch.nn.Parameter(
            self.attn_alpha_init_scaling
            * torch.ones(config.hidden_size, dtype=torch.float32)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / (config.hidden_size**0.5)
        self.mlp_alpha = torch.nn.Parameter(
            self.mlp_alpha_init_scaling
            * torch.ones(config.hidden_size, dtype=torch.float32)
        )

        self.setup(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        if layer_past is not None:
            print(layer_past[0].shape)
        else:
            print("No past")
        # Get eigen learning rates (either static or dynamic)
        attn_rates, mlp_rates = self.get_eigen_rates(hidden_states)

        attention_layer_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        attn_output = self.normalize(attn_output)

        hidden_states = self.normalize(hidden_states + attn_rates * attn_output)

        mlp_output = self.mlp(hidden_states)
        mlp_output = self.post_mlp_dropout(mlp_output)
        mlp_output = self.normalize(mlp_output)

        hidden_states = self.normalize(hidden_states + mlp_rates * mlp_output)

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs
