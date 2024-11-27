from typing import Optional, Tuple
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.gemma.modeling_gemma import (
    GemmaConfig,
    GemmaMLP,
    GemmaRMSNorm,
    GEMMA_ATTENTION_CLASSES,
)


class GatedGemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        layer_idx: int,
        init_weight_std=0.01,
        init_bias_val=2.2,
        sparsity_loss_weight=0.01,
        entropy_loss_weight=0.01,
    ):
        super().__init__()
        self.sparsity_loss_weight = sparsity_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_gate = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp_gate = nn.Linear(config.hidden_size, config.hidden_size)

        nn.init.normal_(self.attn_gate.weight, std=init_weight_std)
        nn.init.constant_(self.attn_gate.bias, init_bias_val)

        nn.init.normal_(self.mlp_gate.weight, std=init_weight_std)
        nn.init.constant_(self.mlp_gate.bias, init_bias_val)

        self.current_attn_gate_output: Optional[torch.Tensor] = None
        self.current_mlp_gate_output: Optional[torch.Tensor] = None

    def get_entropy_loss(self, gate_output: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        entropy_loss = -(
            gate_output * (gate_output + eps).log()
            + (1 - gate_output) * (1 - gate_output + eps).log()
        )

        return entropy_loss

    def get_gate_loss(self) -> torch.Tensor:
        loss = torch.tensor(0)
        if self.current_attn_gate_output is not None:
            loss += (
                self.sparsity_loss_weight * self.current_attn_gate_output.abs().mean()
            )
            loss += self.entropy_loss_weight * self.get_entropy_loss(
                self.current_attn_gate_output
            )
        if self.current_mlp_gate_output is not None:
            loss += (
                self.sparsity_loss_weight * self.current_mlp_gate_output.abs().mean()
            )
            loss += self.entropy_loss_weight * self.get_entropy_loss(
                self.current_mlp_gate_output
            )

        return loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
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
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        gate = torch.sigmoid(self.attn_gate(hidden_states))
        self.current_attn_gate_output = gate
        hidden_states = gate * attn_output + (1 - gate) * residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        gate = torch.sigmoid(self.mlp_gate(hidden_states))
        self.current_mlp_gate_output = gate
        hidden_states = gate * mlp_output + (1 - gate) * residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
