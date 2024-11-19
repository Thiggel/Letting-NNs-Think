from lightning import LightningModule
from typing import Protocol
from transformers import PreTrainedModel
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiment.configs import ModelConfig, DataConfig


class UninterruptedLanguageModelProtocol(Protocol):
    config: ModelConfig
    data_config: DataConfig
    model: PreTrainedModel

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...


class UninterruptedLanguageModel:
    def get_similarity_loss(
        self: UninterruptedLanguageModelProtocol,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        lm_loss = outputs.loss

        if self.config.make_uninterrupted and outputs.hidden_states is not None:
            # Custom loss: make last hidden state similar to next token's first embedded state
            last_hidden_states = outputs.hidden_states[-1][:, :-1, :]
            next_token_embeddings = self.model.get_input_embeddings()(
                batch["input_ids"][:, 1:]
            )
            similarity_loss = F.mse_loss(last_hidden_states, next_token_embeddings)

            self.log(
                f"{mode}_similarity_loss",
                similarity_loss,
                sync_dist=True,
                batch_size=self.data_config.batch_size,
            )

            total_loss = (
                lm_loss + self.config.uninterrupted_loss_weight * similarity_loss
            )
            return total_loss

        return lm_loss or torch.tensor(0.0)

    def get_recurrent_prediction_loss(
        self: UninterruptedLanguageModelProtocol,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        """
        Compute loss based on recurrent predictions using last hidden states.
        Accumulate hidden states during recurrence to allow proper attention.

        Args:
            outputs: Model outputs containing hidden states
            batch: Input batch containing input_ids and other tensors
            mode: Training mode ('train', 'val', or 'test')

        Returns:
            Combined loss from cross entropy of predictions and MSE of hidden states
        """
        if not (
            self.config.make_uninterrupted_with_recurrence
            and outputs.hidden_states is not None
        ):
            return torch.tensor(0.0, device=self.device)

        # Get the last hidden states for all tokens
        last_hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        # Get original embeddings
        original_embeddings = self.model.get_input_embeddings()(
            batch["input_ids"]
        )  # [batch, seq_len, hidden_dim]

        batch_size, seq_length, hidden_dim = last_hidden_states.shape

        all_prediction_losses = []
        all_hidden_state_losses = []

        # For each starting position
        for start_pos in range(
            seq_length - 1
        ):  # -1 because last token can't predict anything
            # Initial hidden states: stack first tokens up to start_pos
            initial_hidden = torch.cat(
                [
                    original_embeddings[:, : start_pos + 1, :],  # original embeddings
                    last_hidden_states[
                        :, start_pos + 1 : start_pos + 2, :
                    ],  # last hidden state of next token
                ],
                dim=1,
            )

            # Prepare variables for recurrence
            current_hidden_sequence = initial_hidden
            position_predictions = []
            position_hidden_states = []

            # Predict all remaining tokens from this position
            remaining_steps = seq_length - (start_pos + 1)

            for step in range(remaining_steps):
                # Use entire accumulated hidden sequence for model input
                transformer_outputs = self.model(
                    inputs_embeds=current_hidden_sequence,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get the last hidden state of the last token as prediction hidden state
                last_token_hidden = transformer_outputs.hidden_states[-1][:, -1:, :]

                # Generate prediction for the next token
                lm_head_output = self.model.lm_head(
                    last_token_hidden
                )  # [batch, 1, vocab_size]
                position_predictions.append(lm_head_output)
                position_hidden_states.append(last_token_hidden)

                # Prepare for next iteration: add the current hidden state to the sequence
                current_hidden_sequence = torch.cat(
                    [current_hidden_sequence, last_token_hidden], dim=1
                )

            if position_predictions:  # Skip if no predictions (last token)
                # Stack predictions and hidden states for this position
                pos_preds = torch.cat(
                    position_predictions, dim=1
                )  # [batch, remaining_steps, vocab_size]
                pos_hidden = torch.cat(
                    position_hidden_states, dim=1
                )  # [batch, remaining_steps, hidden_dim]

                # Get target tokens and embeddings for the remaining sequence
                target_tokens = batch["input_ids"][
                    :, start_pos + 1 : seq_length
                ]  # [batch, remaining_steps]
                target_embeds = original_embeddings[
                    :, start_pos + 1 : seq_length, :
                ]  # [batch, remaining_steps, hidden_dim]

                # Compute losses for this position
                pred_loss = F.cross_entropy(
                    pos_preds.view(-1, pos_preds.size(-1)), target_tokens.view(-1)
                )
                hidden_loss = F.mse_loss(pos_hidden, target_embeds)

                all_prediction_losses.append(pred_loss)
                all_hidden_state_losses.append(hidden_loss)

        # Average losses across all positions
        if all_prediction_losses:  # Check if we have any predictions
            avg_prediction_loss = torch.stack(all_prediction_losses).mean()
            avg_hidden_loss = torch.stack(all_hidden_state_losses).mean()

            # Combine losses with configurable weights
            total_loss = (
                self.config.recurrent_prediction_weight * avg_prediction_loss
                + self.config.recurrent_hidden_state_weight * avg_hidden_loss
            )

            # Log individual losses
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

        return torch.tensor(0.0, device=self.device)
