from tqdm import tqdm
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
    device: torch.device

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...

    def get_similarity_loss(
        self,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor: ...

    def get_recurrent_prediction_loss(
        self,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor: ...


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
            return total_loss + self.get_recurrent_prediction_loss(outputs, batch, mode)

        return lm_loss or torch.tensor(0.0)

    def get_recurrent_prediction_loss(
        self,
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
        if not (
            self.config.make_uninterrupted_with_recurrence
            and outputs.hidden_states is not None
        ):
            return torch.tensor(0.0, device=self.device)

        # Get model configuration details
        pad_token_id = self.tokenizer.pad_token_id
        device = batch["input_ids"].device

        # Create a mask for valid tokens (non-pad tokens)
        padding_mask = batch["input_ids"] != pad_token_id  # [batch, seq_len]
        seq_length = (
            padding_mask.sum(dim=1).max().item()
        )  # Maximum number of non-pad tokens

        # Extract original embeddings
        original_embeddings = self.model.get_input_embeddings()(
            batch["input_ids"]
        )  # [batch, seq_len, hidden_dim]

        # Initialize losses
        all_prediction_losses = []
        all_hidden_state_losses = []

        # Start with the original embeddings for recurrence
        current_sequence = original_embeddings.clone()  # Initial input embeddings
        for step in tqdm(
            range(seq_length - 1), desc="Processing sequence recurrently..."
        ):  # -1 because last token cannot predict further

            # Forward pass through the model
            transformer_outputs = self.model(
                inputs_embeds=current_sequence,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get the last hidden state for each token
            last_hidden_states = transformer_outputs.hidden_states[
                -1
            ]  # [batch, seq_len, hidden_dim]

            # Generate predictions for all tokens
            lm_head_outputs = self.model.lm_head(
                last_hidden_states
            )  # [batch, seq_len, vocab_size]

            # Cross-entropy loss for predictions (mask out pad tokens)
            target_tokens = batch["input_ids"]  # [batch, seq_len]
            pred_loss = F.cross_entropy(
                lm_head_outputs.reshape(
                    -1, lm_head_outputs.size(-1)
                ),  # [batch * seq_len, vocab_size]
                target_tokens.reshape(-1),  # [batch * seq_len]
                reduction="none",
            )
            pred_loss = pred_loss[padding_mask.view(-1)]  # Mask out pad positions
            pred_loss = pred_loss.mean()  # Average loss over valid tokens

            # MSE loss between hidden states and original embeddings (mask out pad tokens)
            target_embeds = original_embeddings  # [batch, seq_len, hidden_dim]
            mse_loss = F.mse_loss(
                last_hidden_states, target_embeds, reduction="none"
            )  # [batch, seq_len, hidden_dim]
            mse_loss = mse_loss[
                padding_mask.unsqueeze(-1).expand_as(mse_loss)
            ]  # Mask out pad positions
            mse_loss = mse_loss.mean()  # Average loss over valid tokens

            # Accumulate losses
            all_prediction_losses.append(pred_loss)
            all_hidden_state_losses.append(mse_loss)

            # Prepare the next sequence for recurrence
            next_hidden_sequence = last_hidden_states.clone()
            next_hidden_sequence = torch.cat(
                [
                    original_embeddings[
                        :, :1, :
                    ],  # Re-add embedding for the first token
                    next_hidden_sequence[:, :-1, :],  # Remove the last hidden state
                ],
                dim=1,
            )
            current_sequence = next_hidden_sequence

        # Finalize losses
        if all_prediction_losses:  # Only if there are valid predictions
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

        return torch.tensor(0.0, device=self.device)
