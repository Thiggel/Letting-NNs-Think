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

    def _forward_without_lm_head(
        self, model: PreTrainedModel, batch: dict[str, torch.Tensor]
    ) -> CausalLMOutputWithPast: ...

    def _get_prediction_loss(
        self,
        last_hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor: ...

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

    def _shift_left(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def _shift_right(self, tensor: torch.Tensor) -> torch.Tensor: ...

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

    def _get_prediction_loss(
        self: UninterruptedLanguageModelProtocol,
        last_hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model.get_output_embeddings()(last_hidden_states)
        prediction_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.reshape(-1)
        )

        return prediction_loss

    def _shift_left(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, 1:]

    def _shift_right(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :-1]

    def checkpointed_forward(
        self: UninterruptedLanguageModelProtocol,
        model: nn.Module,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_without_lm_head(model, sequence, attention_mask)

    def _forward_without_lm_head(
        self: UninterruptedLanguageModelProtocol,
        model: PreTrainedModel,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CausalLMOutputWithPast:
        lm_head = model.get_output_embeddings()
        model.set_output_embeddings(nn.Identity())

        original_dtype = sequence.dtype
        outputs = model(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits.to(original_dtype)

        self.model.set_output_embeddings(lm_head)

        return outputs

    def get_recurrent_prediction_loss(
        self: UninterruptedLanguageModelProtocol,
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
        input_embeddings = self.model.get_input_embeddings()(batch["input_ids"])
        sequence = input_embeddings.clone().detach().requires_grad_(True)

        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        all_prediction_losses = []
        all_hidden_state_losses = []

        seq_len = input_embeddings.shape[1]
        num_steps = min(seq_len - 1, self.config.uninterrupted_recurrence_depth)

        for step in tqdm(
            range(num_steps),
            desc="Processing sequence recurrently...",
        ):
            last_hidden_states = checkpoint(
                self.checkpointed_forward,
                self.model,
                sequence,
                attention_mask,
            )

            if last_hidden_states is None:
                raise ValueError("Model forward pass failed")

            sequence = self._shift_right(last_hidden_states)
            attention_mask = self._shift_right(attention_mask)
            labels = self._shift_left(labels)
            input_embeddings = self._shift_left(input_embeddings)

            prediction_loss = self._get_prediction_loss(sequence, labels)
            similarity_loss = self._get_similarity_loss(sequence, input_embeddings)

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
