import torch
from torch import nn
from typing import Optional, Tuple

from .RecurrenceStrategy import RecurrenceStrategy
from .FixedStepsStrategy import FixedStepsStrategy
from .TimestepEmbedder import TimestepEmbedder
from .exit_classifier_strategy import ExitClassifierStrategy
from .FixedPointStrategy import FixedPointStrategy
from experiment.configs.ModelConfig import ModelConfig


class RecurrentTransformerLayer(nn.Module):
    """
    A transformer layer that can be made recurrent using different strategies.
    Supports fixed steps, adaptive exit classifier, or fixed point iteration.
    """

    def __init__(
        self,
        layer: nn.Module,
        config: ModelConfig,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.layer = layer
        self.strategy = self._create_strategy(config, hidden_size, config.max_steps)
        self.intermediate_outputs = None

    def _create_strategy(
        self, config: ModelConfig, hidden_size: int, max_steps: int
    ) -> RecurrenceStrategy:
        self.timestep_embedder = (
            TimestepEmbedder(hidden_size) if config.use_time_embedding else None
        )

        if config.num_steps is None:
            return FixedStepsStrategy(max_steps, hidden_size, config.use_time_embedding)

        if isinstance(config.num_steps, int):
            return FixedStepsStrategy(
                config.num_steps,
                config.use_time_embedding,
                self.timestep_embedder,
            )

        if config.num_steps == "classifier":
            return ExitClassifierStrategy(config, hidden_size)

        if config.num_steps == "fixed_point":
            return FixedPointStrategy()

        raise ValueError(f"Unknown recurrence mode: {config.num_steps}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Handle special layer requirements
        if hasattr(self.layer, "squeeze_seq_len"):
            hidden_states = self.layer.squeeze_seq_len(hidden_states)
        if hasattr(self.layer, "reset_state"):
            self.layer.reset_state(hidden_states)

        # Clear caching arguments
        kwargs["past_key_value"] = None
        kwargs["use_cache"] = False

        # Run the recurrence strategy
        output = self.strategy.forward(
            hidden_states, attention_mask, self.layer, position_ids, **kwargs
        )

        # Handle special layer requirements
        if hasattr(self.layer, "unsqueeze_seq_len"):
            output.hidden_states = self.layer.unsqueeze_seq_len(output.hidden_states)

        self.intermediate_outputs = output.intermediate_outputs

        return (
            output.hidden_states,
            past_key_value,
            output.exit_probs,
        )
