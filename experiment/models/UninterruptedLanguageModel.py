from tqdm import tqdm
from typing import Protocol
from transformers import PreTrainedModel
from torch.utils.checkpoint import checkpoint
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiment.configs import ModelConfig, DataConfig


class UninterruptedLanguageModelProtocol(Protocol):
    config: ModelConfig
    data_config: DataConfig
    model: PreTrainedModel
    device: torch.device

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...

    def _get_similarity_loss(
        self,
        last_hidden_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor: ...

    def get_recurrent_prediction_loss(
        self,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor: ...

    def shift_left(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def shift_right(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def checkpointed_forward(
        self, module: nn.Module, sequence: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor: ...


class UninterruptedLanguageModel:
    def _get_similarity_loss(
        self: UninterruptedLanguageModelProtocol,
        last_hidden_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        similarity_loss = F.mse_loss(last_hidden_states, next_token_embeddings)

        return self.config.uninterrupted_loss_weight * similarity_loss

    def checkpointed_forward(
        self: UninterruptedLanguageModelProtocol,
        module: nn.Module,
        sequence: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wrapper function for applying checkpointing to a module.
        """

        def forward(sequence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return module(
                inputs_embeds=sequence,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )

        return forward(sequence, labels)

    def _shift_left(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, 1:]

    def _shift_right(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :-1]

    def get_recurrent_prediction_loss(
        self: UninterruptedLanguageModelProtocol,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        """
        Compute loss based on recurrent predictions using last hidden states.
        Accumulate hidden states during recurrence to allow proper attention,
        skipping pad tokens and applying causal masking.

        Args:
            outputs: Model outputs containing hidden states
            batch: Input batch containing input_ids and other tensors
            mode: Training mode ('train', 'val', or 'test')

        Returns:
            Combined loss from cross entropy of predictions and MSE of hidden states
        """
        if not self.config.make_uninterrupted or outputs.hidden_states is None:
            return torch.tensor(0.0, device=self.device)

        input_embeddings = self.model.get_input_embeddings()(batch["input_ids"])
        sequence = input_embeddings.clone()

        labels = batch["labels"]

        all_prediction_losses = []
        all_hidden_state_losses = []

        seq_len = input_embeddings.shape[1]
        num_steps = min(seq_len - 1, self.config.uninterrupted_recurrence_depth)

        for step in tqdm(
            range(num_steps),
            desc="Processing sequence recurrently...",
        ):
            transformer_outputs = checkpoint(
                self.checkpointed_forward, self.model, sequence, batch["labels"]
            )

            if transformer_outputs is None:
                raise ValueError("Model forward pass failed")

            last_hidden_states = transformer_outputs.hidden_states[-1]

            prediction_loss = transformer_outputs.loss

            # For cross entropy loss, HF automatically shifts
            # labels and predictions, so we do not need to shift them
            # at the first step, hence we shift at the end of each step
            labels = self.shift_left(labels)
            sequence = self.shift_right(sequence)

            # For MSE loss, we need to shift the embeddings ourselves
            # from the start
            input_embeddings = self.shift_right(input_embeddings)

            similarity_loss = self._get_similarity_loss(
                last_hidden_states, input_embeddings
            )

            all_prediction_losses.append(prediction_loss)
            all_hidden_state_losses.append(similarity_loss)

        avg_prediction_loss = torch.stack(all_prediction_losses).mean()
        avg_hidden_loss = torch.stack(all_hidden_state_losses).mean()

        # Combine losses with configurable weights
        total_loss = (
            self.config.recurrent_prediction_weight * avg_prediction_loss
            + self.config.recurrent_hidden_state_weight * avg_hidden_loss
        )

        # Log individual losses for tracking
        self.log(
            f"{mode}_recurrent_prediction_loss",
            avg_prediction_loss,
            sync_dist=True,
            batch_size=self.data_config.batch_size,
        )
        self.log(
            f"{mode}_recurrent_hidden_state_loss",
            avg_hidden_loss,
            sync_dist=True,
            batch_size=self.data_config.batch_size,
        )

        return total_loss
