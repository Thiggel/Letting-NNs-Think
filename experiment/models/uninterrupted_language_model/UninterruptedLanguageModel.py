from typing import Protocol, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiment.configs import ModelConfig, DataConfig, UninterruptedMode, FinetuneMode

from experiment.model_evaluator.CustomInference import UninterruptedTransformer

from .GMMHead import GMMHead


class UninterruptedLanguageModelProtocol(Protocol):
    config: ModelConfig
    data_config: DataConfig
    model: PreTrainedModel
    device: torch.device
    uninterrupted_adapter: GMMHead
    tokenizer: PreTrainedTokenizer
    generator: Optional[UninterruptedTransformer]

    def log(
        self,
        name: str,
        value: torch.Tensor,
        sync_dist: bool = False,
        batch_size: int = 1,
    ) -> None: ...

    def _forward(
        self,
        model: PreTrainedModel,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> CausalLMOutputWithPast: ...

    def get_recurrent_prediction_loss(
        self,
        outputs: CausalLMOutputWithPast,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor: ...


class UninterruptedLanguageModel:
    def _uninterruted_setup(self: UninterruptedLanguageModelProtocol) -> None:
        if self.config.uninterrupted_mode == UninterruptedMode.GMM:
            self.uninterrupted_adapter = GMMHead(self.model.config.hidden_size)
            self.generator = UninterruptedTransformer(self, self.tokenizer, alpha=1.0)

    def _forward(
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
            use_cache=False,
        )

        return outputs

    def get_recurrent_prediction_loss(
        self: UninterruptedLanguageModelProtocol,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        input_embeddings = self.model.get_input_embeddings()(batch["input_ids"])
        sequence = input_embeddings.clone()

        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        all_prediction_losses = []
        all_hidden_state_losses = []

        seq_len = input_embeddings.shape[1]
        num_steps = min(seq_len - 1, self.config.uninterrupted_recurrence_depth)

        for i in range(num_steps):
            outputs = self._forward(self.model, sequence, attention_mask, labels)

            last_hidden_states = outputs.hidden_states[-1]
            # projected_states = self.uninterrupted_adapter(last_hidden_states)

            prediction_loss = outputs.loss * self.config.loss_discount_factor**i

            self.log(
                f"{mode}_prediction_loss_step_{i}",
                prediction_loss,
                sync_dist=True,
                batch_size=self.data_config.batch_size,
            )

            if self.config.uninterrupted_mode == UninterruptedMode.GMM:
                gmm_loss = (
                    self.uninterrupted_adapter.loss(
                        last_hidden_states[:, :-1],
                        input_embeddings[:, 1:],
                    )
                    * self.config.loss_discount_factor**i
                )

                self.log(
                    f"{mode}_gmm_loss_step_{i}",
                    gmm_loss,
                    sync_dist=True,
                    batch_size=self.data_config.batch_size,
                )

                all_hidden_state_losses.append(gmm_loss)

            # sequence = torch.cat(
            #    [
            #        input_embeddings[:, :1],
            #        projected_states[:, :-1],
            #    ],
            #    dim=1,
            # )

            if not (self.config.finetune_mode == FinetuneMode.FROZEN and i == 0):
                all_prediction_losses.append(prediction_loss)

        avg_prediction_loss = 0
        if len(all_prediction_losses) != 0:
            avg_prediction_loss = torch.stack(all_prediction_losses).mean()
            self.log(
                f"{mode}_recurrent_prediction_loss",
                avg_prediction_loss,
                sync_dist=True,
                batch_size=self.data_config.batch_size,
            )

        avg_hidden_loss = 0
        if self.config.uninterrupted_mode == UninterruptedMode.GMM:
            avg_hidden_loss = torch.stack(all_hidden_state_losses).mean()

            self.log(
                f"{mode}_recurrent_hidden_state_loss",
                avg_hidden_loss,
                sync_dist=True,
                batch_size=self.data_config.batch_size,
            )

        total_loss = (
            self.config.recurrent_prediction_weight * avg_prediction_loss
            + self.config.recurrent_hidden_state_weight * avg_hidden_loss
        )

        return total_loss
