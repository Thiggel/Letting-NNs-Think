from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class RecurrenceOutput:
    """Container for recurrence step outputs"""

    hidden_states: torch.Tensor
    exit_probs: Optional[torch.Tensor] = None
    intermediate_outputs: Optional[list[torch.Tensor]] = None
