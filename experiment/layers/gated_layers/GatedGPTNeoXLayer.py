from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXMLP,
    GPT_NEOX_ATTENTION_CLASSES,
)


class GatedGPTNeoXLayer(nn.Module):
    def __init__(
        self,
        config,
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

        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.self_attn = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = GPTNeoXMLP(config)

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
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        attention_layer_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
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

        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))

        # Attention
        gate = torch.sigmoid(self.attn_gate(hidden_states))
        self.current_attn_gate_output = gate
        hidden_states = gate * attn_output + (1 - gate) * residual

        # MLP
        residual = hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)

        gate = torch.sigmoid(self.mlp_gate(hidden_states))
        self.current_mlp_gate_output = gate
        hidden_states = gate * mlp_output + (1 - gate) * residual

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs
