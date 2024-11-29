from typing import Optional, Tuple
import torch
from transformers.cache_utils import Cache
from transformers.models.gemma.modeling_gemma import (
    GemmaConfig,
    GemmaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

from ..CanNormalize import CanNormalize

logger = logging.get_logger(__name__)


class NormalizedGemmaSdpaAttention(GemmaAttention, CanNormalize):
    """
    Gemma attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GemmaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = 1.0 / (config.hidden_size**0.5)
        self.sqk_query = torch.nn.Parameter(
            self.sqk_init_scaling
            * torch.ones(
                self.config.num_attention_heads * self.head_dim, dtype=torch.float32
            )
        )
        self.sqk_key = torch.nn.Parameter(
            self.sqk_init_scaling
            * torch.ones(
                self.config.num_key_value_heads * self.head_dim,
                dtype=torch.float32,
            )
        )

    def normalize_weights(self):
        self.q_proj.weight.data.copy_(self.normalize(self.q_proj.weight.data))
        self.k_proj.weight.data.copy_(self.normalize(self.k_proj.weight.data))
        self.v_proj.weight.data.copy_(self.normalize(self.v_proj.weight.data))
        self.o_proj.weight.data.copy_(self.normalize(self.o_proj.weight.data, 0))

    # Adapted from GemmaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "GemmaModel is using GemmaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        sqk_query = (
            self.sqk_query * (self.sqk_init_value / self.sqk_init_scaling)
        ).view(1, self.config.num_attention_heads, 1, self.head_dim)

        sqk_key = (self.sqk_key * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, self.config.num_key_value_heads, 1, self.head_dim
        )

        query_states = sqk_query * self.normalize(query_states)
        key_states = sqk_key * self.normalize(key_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.hidden_size**0.5,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
