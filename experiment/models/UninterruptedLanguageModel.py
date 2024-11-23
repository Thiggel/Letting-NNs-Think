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
        self,
        model: PreTrainedModel,
        batch: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> CausalLMOutputWithPast: ...

    def _get_prediction_loss(
        self,
        last_hidden_states: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def _get_similarity_loss(
        self,
        last_hidden_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
        mask: torch.Tensor,
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
        mask: torch.Tensor,
    ) -> torch.Tensor:
        similarity_loss = (
            F.mse_loss(last_hidden_states, next_token_embeddings, reduction="none")
            * mask.unsqueeze(-1)
        ).sum() / mask.unsqueeze(-1).repeat(1, 1, last_hidden_states.shape[-1]).sum()

        print(similarity_loss)

        return self.config.uninterrupted_loss_weight * similarity_loss

    def _shift_left(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, 1:]

    def _shift_right(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :-1]

    def checkpointed_forward(
        self: UninterruptedLanguageModelProtocol,
        model: nn.Module,
        sequence: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_without_lm_head(model, sequence, attention_mask, labels)

    def _forward_without_lm_head(
        self: UninterruptedLanguageModelProtocol,
        model: PreTrainedModel,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> CausalLMOutputWithPast:
        outputs = model(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs

    def get_recurrent_prediction_loss(
        self: UninterruptedLanguageModelProtocol,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        input_embeddings = self.model.base_model.model.model.embed_tokens(
            batch["input_ids"]
        )
        sequence = input_embeddings.clone()

        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        all_prediction_losses = []
        all_hidden_state_losses = []

        seq_len = input_embeddings.shape[1]
        num_steps = min(seq_len - 1, self.config.uninterrupted_recurrence_depth)

        for _ in range(num_steps):
            outputs = checkpoint(
                self.checkpointed_forward,
                self.model,
                sequence,
                attention_mask,
                labels,
            )

            if outputs is None:
                raise ValueError("Model forward pass failed")

            last_hidden_states = outputs.hidden_states[-1]

            prediction_loss = outputs.loss
            similarity_loss = self._get_similarity_loss(
                last_hidden_states, input_embeddings, attention_mask
            )

            sequence = self._shift_right(last_hidden_states)
            attention_mask = self._shift_right(attention_mask)
            labels = self._shift_left(labels)
            input_embeddings = self._shift_left(input_embeddings)

            # TODO: think through again:
            # sequence = torch.cat(
            #    [
            #        input_embeddings[:, :1],
            #        sequence,
            #    ],
            #    dim=1,
            # )

            all_prediction_losses.append(prediction_loss)
            all_hidden_state_losses.append(similarity_loss)

        avg_prediction_loss = torch.stack(all_prediction_losses).mean()
        avg_hidden_loss = torch.stack(all_hidden_state_losses).mean()

        total_loss = (
            self.config.recurrent_prediction_weight * avg_prediction_loss
            + self.config.recurrent_hidden_state_weight * avg_hidden_loss
        )

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
