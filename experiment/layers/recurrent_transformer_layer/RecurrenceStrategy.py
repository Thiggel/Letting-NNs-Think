from abc import ABC, abstractmethod
from typing import Optional
import torch

from .TransformerLayerProtocol import TransformerLayerProtocol
from .RecurrenceOutput import RecurrenceOutput


class RecurrenceStrategy(ABC):
    """Abstract base class for different recurrence strategies"""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RecurrenceOutput: ...
