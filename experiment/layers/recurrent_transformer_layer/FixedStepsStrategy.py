import torch
from typing import Optional

from .TransformerLayerProtocol import TransformerLayerProtocol
from .RecurrenceOutput import RecurrenceOutput
from .RecurrenceStrategy import RecurrenceStrategy


class FixedStepsStrategy(RecurrenceStrategy):
    """Strategy for fixed number of recurrence steps"""

    def __init__(self, num_steps: int, use_time_embedding: bool = False):
        self.num_steps = num_steps
        self.use_time_embedding = use_time_embedding

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RecurrenceOutput:
        intermediate_outputs = []

        for step in range(self.num_steps):
            if self.use_time_embedding:
                hidden_states = hidden_states + (step + 1)

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )

            if step < self.num_steps - 1:
                intermediate_outputs.append(hidden_states.clone())

        return RecurrenceOutput(
            hidden_states=hidden_states, intermediate_outputs=intermediate_outputs
        )
