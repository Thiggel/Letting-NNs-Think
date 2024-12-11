from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.utils import logging
from transformers.cache_utils import Cache

from ..NormalizedDecoderLayer import NormalizedDecoderLayer
from .NormalizedLlamaMLP import NormalizedLlamaMLP
from .NormalizedLlamaSdpaAttention import NormalizedLlamaSdpaAttention

logger = logging.get_logger(__name__)


class NormalizedLlamaDecoderLayer(nn.Module, NormalizedDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        use_dynamic_rates: bool = False,
        use_momentum: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_dynamic_rates = use_dynamic_rates
        self.use_momentum = use_momentum

        # Core components
        self.self_attn = NormalizedLlamaSdpaAttention(
            config=config, layer_idx=layer_idx
        )

        self.mlp = NormalizedLlamaMLP(config)

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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        attn_rates, mlp_rates = self.get_eigen_rates(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = F.normalize(hidden_states, dim=-1)

        # Calculate attention update with optional momentum
        attn_delta = hidden_states - residual
        if self.use_momentum:
            attn_delta = self.update_momentum(attn_delta, self.attn_momentum)

        hidden_states = F.normalize(residual + attn_rates * attn_delta, dim=-1)

        # Fully Connected
        residual = hidden_states
        hidden_states = F.normalize(self.mlp(hidden_states), dim=-1)

        # Calculate MLP update with optional momentum
        mlp_delta = hidden_states - residual
        if self.use_momentum:
            mlp_delta = self.update_momentum(mlp_delta, self.mlp_momentum)

        hidden_states = F.normalize(residual + mlp_rates * mlp_delta, dim=-1)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
