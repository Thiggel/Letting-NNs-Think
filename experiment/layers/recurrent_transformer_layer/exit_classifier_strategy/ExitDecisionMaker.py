import torch
from torch import nn


class ExitDecisionMaker:
    """Handles exit decisions and state updates"""

    def __init__(self, hidden_size: int):
        self.exit_classifier = nn.Linear(hidden_size, 1)

    def compute_exit_decisions(
        self, classifier_inputs: torch.Tensor, inference_mode: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        exit_logits = self.exit_classifier(classifier_inputs)
        exit_probs = torch.sigmoid(exit_logits)

        if inference_mode:
            exit_decisions = exit_probs > 0.5
        else:
            exit_decisions = torch.bernoulli(exit_probs).bool()

        return exit_probs, exit_decisions.squeeze(-1)
