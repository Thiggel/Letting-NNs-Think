import torch
from typing import Optional, Tuple
from packaging import version
from transformers.utils import (
    get_torch_version,
    logging,
)
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoXAttention,
    apply_rotary_pos_emb,
)

logger = logging.get_logger(__name__)


class NormalizedGPTNeoXSdpaAttention(GPTNeoXAttention):
    """
    GPTNeoX attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GPTNeoXAttention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(
            get_torch_version()
        ) < version.parse("2.2.0")

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = 1.0 / (config.hidden_size**0.5)
        self.sqk = torch.nn.Parameter(
            self.sqk_init_scaling
            * torch.ones(
                self.config.num_attention_heads * self.head_dim, dtype=torch.float32
            )
        )

    def normalize_weights(self):
        # Split the weight matrix along the first dimension into Q, K, V portions
        q_weight, k_weight, v_weight = self.query_key_value.weight.data.chunk(3, dim=0)

        # Normalize each portion separately
        q_normalized = self.normalize(q_weight)
        k_normalized = self.normalize(k_weight)
        v_normalized = self.normalize(v_weight)

        # Concatenate back together
        self.query_key_value.weight.data.copy_(
            torch.cat([q_normalized, k_normalized, v_normalized], dim=0)
        )
        self.dense.weight.data.copy_(self.normalize(self.dense.weight.data, 0))

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
    ):
        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        bsz, q_len = query.shape[:2]

        query = query.view(
            bsz, q_len, self.num_attention_heads, self.head_size
        ).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_attention_heads, self.head_size).transpose(
            1, 2
        )
        value = value.view(
            bsz, q_len, self.num_attention_heads, self.head_size
        ).transpose(1, 2)

        sqk = (self.sqk_query * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, self.config.num_attention_heads, 1, self.head_dim
        )

        query = sqk * self.normalize(query)
        key = sqk * self.normalize(key)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value, position_ids)
        else:
            cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

        return query, key, value, layer_past

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
    ):
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`GPTNeoXSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if (
            self.require_contiguous_qkv
            and query.device.type == "cuda"
            and attention_mask is not None
        ):
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.dense(attn_output)

        return attn_output, present, None


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores
