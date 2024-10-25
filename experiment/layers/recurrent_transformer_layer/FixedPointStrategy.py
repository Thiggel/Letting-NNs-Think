import torch
from typing import Optional
from torchdeq import get_deq, reset_deq

from .TransformerLayerProtocol import TransformerLayerProtocol
from .RecurrenceOutput import RecurrenceOutput
from .RecurrenceStrategy import RecurrenceStrategy


class FixedPointStrategy(RecurrenceStrategy):
    """Strategy using fixed point iteration to determine convergence"""

    def __init__(self):
        self.recurrence = get_deq(f_solver="fixed_point_iter")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RecurrenceOutput:
        reset_deq(self.recurrence)

        def f(prev_states: torch.Tensor) -> torch.Tensor:
            outputs = layer(
                prev_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )
            outputs[torch.isnan(outputs)] = 0
            return outputs

        fixed_points, _ = self.recurrence(f, hidden_states, tol=1e-2)
        return RecurrenceOutput(hidden_states=fixed_points[-1])
