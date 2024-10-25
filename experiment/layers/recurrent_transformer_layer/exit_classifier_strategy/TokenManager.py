import torch
from typing import Protocol


class TokenManager(Protocol):
    """Protocol for different token management strategies"""

    def prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor: ...

    def get_classifier_inputs(
        self, hidden_states: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor: ...

    def finalize_hidden_states(
        self, hidden_states: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor: ...
