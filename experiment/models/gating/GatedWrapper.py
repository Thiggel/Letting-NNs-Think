from torch import nn
import torch
from typing import Any, Optional, Union

from transformers import Cache

from experiment.configs.ModelConfig import ModelConfig

from .GateLayer import GateLayer


class GatedWrapper(nn.Module):
    """Wrapper that adds gating to any module"""

    def __init__(
        self,
        module: nn.Module,
        gate: GateLayer,
        config: ModelConfig,
        layer_idx: int,
        module_name: str,
        parent: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.module = module
        self.gate = gate
        self.current_gate_value: Optional[torch.Tensor] = None
        self.current_percent_tokens_skipped = 0.0
        self.current_token_importance: Optional[torch.Tensor] = None
        self.past_percent_skipped: list[float] = []

        object.__setattr__(self, "parent", parent)

    def update_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Cache,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        current_length = past_key_value.get_seq_length(self.module.layer_idx)

        # Create k/v states for the new token position
        value_states = key_states = torch.zeros(
            (
                batch_size,
                self.module.config.num_key_value_heads,
                seq_len,  # Just for the new token(s)
                self.module.config.head_dim,
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Get positions for the new token(s)
        if position_ids is None:
            # Start from current cache length
            position_ids = torch.arange(
                current_length, current_length + seq_len, device=hidden_states.device
            ).unsqueeze(0)

        cos, sin = position_embeddings

        cache_kwargs = {"sin": sin, "cos": cos}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position

        # Update cache with zero states for this position
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.module.layer_idx, cache_kwargs
        )

        return key_states, value_states

    def get_attn_token_importance(self) -> Optional[torch.Tensor]:
        if hasattr(self.parent, "attn"):
            attn_layer = self.parent.attn
        elif hasattr(self.parent, "self_attn"):
            attn_layer = self.parent.self_attn
        else:
            raise ValueError("No attention layer found in parent module")

        return attn_layer.current_token_importance

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        Union[int, torch.Tensor],
        tuple[Union[int, torch.Tensor], Optional[torch.Tensor], Optional[Cache]],
    ]:
        gate_value = self.gate(hidden_states)  # [batch, seq_len, hidden_dim]
        self.current_gate_value = gate_value

        any_skipped_mod_type_is_contained_in_name = any(
            mod_type in self.module_name for mod_type in self.config.skip_module_types
        )

        if (
            self.config.skip_modules
            and any_skipped_mod_type_is_contained_in_name
            and not (
                self.config.only_skip_every_second_layer and self.layer_idx % 2 == 0
            )
        ):
            batch_size, seq_len, hidden_dim = hidden_states.shape

            if self.config.randomly_skip:
                token_importance = (
                    torch.rand(batch_size, seq_len, device=hidden_states.device)
                    > self.config.percent_randomly_skip
                ).float()

            elif (
                self.config.always_skip_layer != -1
                and self.layer_idx == self.config.always_skip_layer
            ):
                token_importance = torch.zeros(
                    batch_size, seq_len, device=hidden_states.device
                )

            elif (
                self.config.skip_entire_layer_based_on_attn
                and "mlp" in self.module_name
            ):
                token_importance = self.get_attn_token_importance()

                assert token_importance is not None

            else:
                token_importance = gate_value.mean(dim=-1)  # [batch, seq_len]

            self.current_token_importance = token_importance

            last_seq_pos_token_importance = token_importance[:, -1]
            process_mask = (
                (last_seq_pos_token_importance > self.config.skip_threshold)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )  # [batch, 1, 1]

            num_skip = process_mask.logical_not().sum().item()
            self.current_percent_tokens_skipped = num_skip / batch_size
            self.past_percent_skipped.append(self.current_percent_tokens_skipped)

            is_attn_layer = hasattr(self.module, "rotary_emb")

            # For attention layers, zero out skipped tokens instead of filtering
            if is_attn_layer:
                # Zero out the hidden states for tokens we want to skip
                zeros = torch.zeros_like(hidden_states)
                effective_hidden_states = torch.where(
                    process_mask.expand_as(hidden_states), hidden_states, zeros
                )

                # Pass through the full tensor
                module_output = self.module(effective_hidden_states, *args, **kwargs)

                if isinstance(module_output, tuple):
                    processed_output = module_output[0]
                    if self.config.actually_gate:
                        processed_output = gate_value * processed_output
                    return (processed_output,) + module_output[1:]
                else:
                    if self.config.actually_gate:
                        module_output = gate_value * module_output
                    return module_output

            # For MLP layers, continue with the original filtering approach
            if self.current_percent_tokens_skipped == 1.0:
                return 0

            # Process only tokens above threshold
            tokens_to_process = hidden_states[
                process_mask.expand_as(hidden_states)
            ].view(-1, seq_len, hidden_dim)

            num_processed = tokens_to_process.shape[0]
            assert batch_size - num_processed == num_skip

            module_output = self.module(tokens_to_process, *args, **kwargs)

            output = hidden_states
            if isinstance(module_output, tuple):
                processed_output = module_output[0]
                if self.config.actually_gate:
                    processed_output = (
                        gate_value[process_mask.expand_as(gate_value)]
                        * processed_output
                    )

                processed_output = processed_output.view(-1, seq_len, hidden_dim)
                assert processed_output.shape == (num_processed, seq_len, hidden_dim)
                output[process_mask.squeeze(1).squeeze(1)] = processed_output
                return (output,) + module_output[1:]
            else:
                if self.config.actually_gate:
                    module_output = (
                        gate_value[process_mask.expand_as(gate_value)] * module_output
                    )

                module_output = module_output.view(-1, seq_len, hidden_dim)
                assert module_output.shape == (num_processed, seq_len, hidden_dim)
                output[process_mask.squeeze(1).squeeze(1)] = module_output
                return output

        # Normal processing
        module_output = self.module(hidden_states, *args, **kwargs)
        if isinstance(module_output, tuple):
            main_output = module_output[0]
            gated_output = (
                gate_value * main_output if self.config.actually_gate else main_output
            )
            return (gated_output,) + module_output[1:]
        else:
            return (
                gate_value * module_output
                if self.config.actually_gate
                else module_output
            )
