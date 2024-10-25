from typing import Optional, Protocol
import torch


class TransformerLayerProtocol(Protocol):
    """Protocol defining the interface expected from the transformer layer"""

    def __call__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor: ...
