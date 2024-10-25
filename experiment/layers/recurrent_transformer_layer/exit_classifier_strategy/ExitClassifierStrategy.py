import torch
from torch import nn
from typing import Optional, Protocol

from experiment.configs import ModelConfig

from ..TransformerLayerProtocol import TransformerLayerProtocol
from ..RecurrenceOutput import RecurrenceOutput
from ..RecurrenceStrategy import RecurrenceStrategy

from .ExitTokenManager import ExitTokenManager
from .DirectTokenManager import DirectTokenManager
from .ExitDecisionMaker import ExitDecisionMaker
from .TokenManager import TokenManager


class ExitClassifierStrategy(RecurrenceStrategy):
    """Strategy using an exit classifier to determine when to stop"""

    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.token_manager: TokenManager = (
            ExitTokenManager() if config.use_exit_tokens else DirectTokenManager()
        )
        self.decision_maker = ExitDecisionMaker(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RecurrenceOutput:
        inference_mode = bool((attention_mask.sum() == attention_mask.shape[1]).item())

        # Prepare inputs
        hidden_states = self.token_manager.prepare_hidden_states(hidden_states)
        attention_mask = self.token_manager.prepare_attention_mask(
            attention_mask, inference_mode
        )

        if inference_mode:
            return self._forward_inference(
                hidden_states, attention_mask, layer, **kwargs
            )
        else:
            return self._forward_train(
                hidden_states, attention_mask, layer, position_ids, **kwargs
            )

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RecurrenceOutput:
        batch_size, seq_len, _ = hidden_states.shape
        seq_len_original = seq_len // 2 if self.config.use_exit_tokens else seq_len

        exit_probs = torch.zeros(
            batch_size, seq_len_original, device=hidden_states.device
        )
        exit_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
        )
        frozen_states = torch.zeros_like(hidden_states)
        intermediate_outputs = []

        for step in range(self.config.max_steps):
            if self.config.use_time_embedding:
                hidden_states = hidden_states + (step + 1)

            # Forward pass through layer
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )

            # Get exit decisions
            classifier_inputs = self.token_manager.get_classifier_inputs(
                hidden_states, False
            )
            exit_prob, exit_decisions = self.decision_maker.compute_exit_decisions(
                classifier_inputs, False
            )

            # Update states
            if self.config.use_exit_tokens:
                exit_mask[:, 1::2] = exit_mask[:, 1::2] | exit_decisions
                exit_mask[:, ::2] = exit_mask[:, 1::2]
            else:
                exit_mask = exit_mask | exit_decisions

            frozen_states = torch.where(
                exit_mask.unsqueeze(-1), hidden_states.detach(), frozen_states
            )
            hidden_states = torch.where(
                exit_mask.unsqueeze(-1), frozen_states, hidden_states
            )

            # Update exit probabilities
            if self.config.use_exit_tokens:
                exit_probs = torch.where(
                    exit_mask[:, 1::2], exit_probs, exit_prob.squeeze(-1)
                )
            else:
                exit_probs = torch.where(exit_mask, exit_probs, exit_prob.squeeze(-1))

            if step < self.config.max_steps - 1:
                intermediate_outputs.append(hidden_states.clone())

        hidden_states = self.token_manager.finalize_hidden_states(hidden_states, False)
        return RecurrenceOutput(
            hidden_states=hidden_states,
            exit_probs=exit_probs,
            intermediate_outputs=intermediate_outputs,
        )

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: TransformerLayerProtocol,
        **kwargs
    ) -> RecurrenceOutput:
        batch_size, seq_len, _ = hidden_states.shape
        exit_prob = torch.tensor(0.0)
        exit_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
        )

        for step in range(self.config.max_steps):
            if self.config.use_time_embedding:
                hidden_states = hidden_states + (step + 1)

            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, **kwargs
            )

            classifier_inputs = self.token_manager.get_classifier_inputs(
                hidden_states, True
            )
            exit_prob, exit_decision = self.decision_maker.compute_exit_decisions(
                classifier_inputs, True
            )

            if exit_prob > self.config.exit_threshold:
                exit_mask[:, -1] = True
                break

            hidden_states = torch.where(
                exit_mask.unsqueeze(-1), hidden_states.detach(), hidden_states
            )

        hidden_states = self.token_manager.finalize_hidden_states(hidden_states, True)
        return RecurrenceOutput(hidden_states=hidden_states, exit_probs=exit_prob)
