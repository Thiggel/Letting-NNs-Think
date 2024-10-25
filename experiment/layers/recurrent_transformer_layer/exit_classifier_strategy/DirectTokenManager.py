import torch

from .TokenManager import TokenManager


class DirectTokenManager(TokenManager):
    """Manages hidden states directly without exit tokens"""

    def prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor:
        return attention_mask

    def get_classifier_inputs(
        self, hidden_states: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor:
        return hidden_states[:, -1:] if inference_mode else hidden_states

    def finalize_hidden_states(
        self, hidden_states: torch.Tensor, inference_mode: bool
    ) -> torch.Tensor:
        return hidden_states
