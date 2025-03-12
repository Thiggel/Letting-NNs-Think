from torch import nn
import torch
from typing import Any, Optional, Union

from experiment.configs.ModelConfig import ModelConfig
from experiment.configs.GatingConfig import GatingMode


class GatedWrapper(nn.Module):
    """Wrapper that adds gating to any module.
    The decision to skip processing is based solely on the current hidden state's gate value.
    When self.config.only_skip_every_second_layer is True, layers that should not be skipped
    (e.g. first layers) will process all tokens, and skipping statistics are recorded as 0% skipped.
    """

    def __init__(
        self,
        module: nn.Module,
        gate: nn.Module,
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

        self.global_step = 1

        object.__setattr__(self, "parent", parent)

    def get_attn_token_importance(self) -> Optional[torch.Tensor]:
        if hasattr(self.parent, "attn"):
            attn_layer = self.parent.attn
        elif hasattr(self.parent, "self_attn"):
            attn_layer = self.parent.self_attn
        else:
            raise ValueError("No attention layer found in parent module")
        return attn_layer.current_token_importance

    def get_threshold(self) -> float:
        if self.config.increasing_threshold:
            delta_threshold = self.config.skip_threshold - self.config.start_threshold
            current_threshold = self.config.start_threshold + (
                self.global_step / self.config.num_increasing_steps * delta_threshold
            )

            return min(
                current_threshold,
                self.config.skip_threshold,
            )

        return self.config.skip_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Union[
        Union[int, torch.Tensor],
        tuple[Union[int, torch.Tensor], Optional[torch.Tensor], Optional[Any]],
    ]:
        # Compute the gate values: shape [B, L, hidden_dim]
        gate_value = self.gate(hidden_states)
        self.current_gate_value = gate_value
        # Compute per-token importance as the mean over hidden dim: shape [B, L]
        token_importance = gate_value.mean(dim=-1)
        self.current_token_importance = token_importance

        # If we are in a layer that should always be processed (when only_skip_every_second_layer is True)
        if self.config.only_skip_every_second_layer and (self.layer_idx % 2 == 0):
            self.current_percent_tokens_skipped = 0.0
            self.past_percent_skipped.append(0.0)
            module_output = self.module(hidden_states, *args, **kwargs)
            if isinstance(module_output, tuple):
                main_output = module_output[0]
                gated_output = (
                    (gate_value * main_output)
                    if self.config.actually_gate
                    else main_output
                )
                return (gated_output,) + module_output[1:]
            else:
                return (
                    (gate_value * module_output)
                    if self.config.actually_gate
                    else module_output
                )

        # Otherwise, if skipping is enabled for this layer:
        any_skipped_mod_type = any(
            mod_type in self.module_name for mod_type in self.config.skip_module_types
        )
        if self.config.skip_modules and any_skipped_mod_type:
            # Optionally override token_importance based on config:
            if self.config.randomly_skip:
                token_importance = (
                    torch.rand_like(token_importance)
                    > self.config.percent_randomly_skip
                ).float()
            elif (
                self.config.always_skip_layer != -1
                and self.layer_idx == self.config.always_skip_layer
            ):
                token_importance = torch.zeros_like(token_importance)
            elif (
                self.config.skip_entire_layer_based_on_attn
                and "mlp" in self.module_name
            ):
                token_importance = self.get_attn_token_importance()

            # Compute process_mask only once:
            threshold = self.get_threshold()
            process_mask = (token_importance > threshold).unsqueeze(-1)

            batch_size, seq_len, hidden_dim = hidden_states.shape
            num_skipped = (~(token_importance > threshold)).sum().item()
            total_tokens = batch_size * seq_len
            self.current_percent_tokens_skipped = num_skipped / total_tokens
            self.past_percent_skipped.append(self.current_percent_tokens_skipped)

            is_attn_layer = "attn" in self.module_name

            if is_attn_layer:
                # For attention layers, zero out tokens not meeting the threshold.
                zeros = torch.zeros_like(hidden_states)
                effective_hidden_states = torch.where(
                    process_mask.expand_as(hidden_states), hidden_states, zeros
                )
                module_output = self.module(effective_hidden_states, *args, **kwargs)

                # Get the past_key_value if it exists
                past_kv = kwargs.get("past_key_value")
                if past_kv is not None and not process_mask.all():
                    # Get previous layer's cache
                    prev_layer_cache = past_kv[
                        self.layer_idx - 1
                    ]  # This returns (k, v) tuple

                    if (
                        isinstance(module_output, tuple)
                        and len(module_output) > 1
                        and module_output[1] is not None
                    ):
                        current_layer_cache = module_output[1][
                            self.layer_idx
                        ]  # Get current layer cache
                        skip_mask = ~process_mask.expand(
                            -1, -1, current_layer_cache[0].size(-1)
                        )

                        # Copy k,v from previous layer for skipped tokens
                        current_layer_cache[0][skip_mask] = prev_layer_cache[0][
                            skip_mask
                        ]
                        current_layer_cache[1][skip_mask] = prev_layer_cache[1][
                            skip_mask
                        ]

                if isinstance(module_output, tuple):
                    processed_output = module_output[0]
                    if self.config.actually_gate:
                        processed_output = gate_value * processed_output
                    return (processed_output,) + module_output[1:]
                else:
                    if self.config.actually_gate:
                        module_output = gate_value * module_output
                    return module_output
            else:
                # For MLP layers, process only the tokens that pass the threshold.
                if process_mask.sum() == 0:
                    return torch.zeros_like(hidden_states)
                tokens_to_process = hidden_states[process_mask.squeeze(-1)]
                module_output = self.module(tokens_to_process, *args, **kwargs)
                output = hidden_states.clone()
                if isinstance(module_output, tuple):
                    processed_output = module_output[0]
                    if self.config.actually_gate:
                        processed_output = (
                            gate_value[process_mask.expand_as(gate_value)]
                            * processed_output
                        )
                    processed_output = processed_output.view(-1, hidden_dim)
                    output[process_mask.squeeze(-1)] = processed_output
                    return (output,) + module_output[1:]
                else:
                    if self.config.actually_gate:
                        module_output = (
                            gate_value[process_mask.expand_as(gate_value)]
                            * module_output
                        )
                    module_output = module_output.view(-1, hidden_dim)
                    output[process_mask.squeeze(-1)] = module_output
                    return output

        # If no skipping is applied at all:
        if (
            self.config.gating_mode == GatingMode.BEFORE_MODULE
            and self.config.actually_gate
        ):
            hidden_states = gate_value * hidden_states

        module_output = self.module(hidden_states, *args, **kwargs)

        if isinstance(module_output, tuple):
            main_output = module_output[0]
            gated_output = (
                (gate_value * main_output)
                if self.config.actually_gate
                and self.config.gating_mode == GatingMode.AFTER_MODULE
                else main_output
            )
            return (gated_output,) + module_output[1:]
        else:
            return (
                (gate_value * module_output)
                if self.config.actually_gate
                and self.config.gating_mode == GatingMode.AFTER_MODULE
                else module_output
            )
