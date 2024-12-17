import torch
from typing import Optional

from .TimestepEmbedder import TimestepEmbedder
from .TransformerLayerProtocol import TransformerLayerProtocol
from .RecurrenceOutput import RecurrenceOutput
from .RecurrenceStrategy import RecurrenceStrategy


class FixedStepsStrategy(RecurrenceStrategy):
    """Strategy for fixed number of recurrence steps"""

    def __init__(
        self,
        num_steps: int,
        use_time_embedding: bool = False,
        timestep_embedder: Optional[TimestepEmbedder] = None,
    ):
        self.num_steps = num_steps
        self.use_time_embedding = use_time_embedding
        self.timestep_embedder = timestep_embedder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        position_ids: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> RecurrenceOutput:
        intermediate_outputs = []

        for step in range(self.num_steps):
            if self.use_time_embedding:
                hidden_states = self.timestep_embedder(hidden_states, step)

            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )[0:2]

            print("present222", present)

            if step < self.num_steps - 1:
                intermediate_outputs.append(hidden_states.clone())

        return RecurrenceOutput(
            hidden_states=hidden_states,
            intermediate_outputs=intermediate_outputs,
            present=present,
        )
