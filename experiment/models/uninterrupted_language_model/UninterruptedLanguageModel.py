from typing import Protocol, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiment.configs import ModelConfig, DataConfig, UninterruptedMode, FinetuneMode

from .UninterruptedLanguageModelInference import UninterruptedLanguageModelInference

from .GMMHead import GMMHead


class UninterruptedLanguageModelProtocol(Protocol):
    config: ModelConfig
    data_config: DataConfig
    model: PreTrainedModel
    device: torch.device
    uninterrupted_adapter: GMMHead
    tokenizer: PreTrainedTokenizer
    _generator: Optional[UninterruptedLanguageModelInference]
    lm_heads: Optional[nn.ModuleList]

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

    def normalize_hidden_state(
        self,
        hidden_state: torch.Tensor,
        embedding_norm: Optional[float] = None,
    ) -> torch.Tensor: ...

    def loss(
        self,
        last_hidden_state: torch.Tensor,
        labels: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor: ...


class UninterruptedLanguageModel:
    def _uninterruted_setup(self: UninterruptedLanguageModelProtocol) -> None:
        if self.config.uninterrupted_mode == UninterruptedMode.GMM:
            self.uninterrupted_adapter = GMMHead(self.model.config.hidden_size)
            self._generator = UninterruptedLanguageModelInference(
                self, self.tokenizer, alpha=1.0
            )
        elif self.config.uninterrupted_mode == UninterruptedMode.DIRECT:
            self.uninterrupted_adapter = nn.Identity()
            self._generator = UninterruptedLanguageModelInference(
                self, self.tokenizer, alpha=1.0, use_adapter=False
            )

        if self.config.different_lm_head_per_step:

            self.lm_heads: nn.ModuleList = nn.ModuleList()

            for _ in range(self.config.uninterrupted_recurrence_depth - 1):
                lm_head_params = self.model.get_output_embeddings().weight.data
                lm_head = nn.Linear(
                    self.model.config.hidden_size, self.model.config.vocab_size
                )
                lm_head.weight.data.copy_(lm_head_params)
                self.lm_heads.append(lm_head)

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

    def normalize_hidden_state(
        self: UninterruptedLanguageModelProtocol, hidden_state, embedding_norm=None
    ):
        """Project hidden state closer to embedding manifold"""
        if embedding_norm is None:
            embedding_norm = (
                self.model.get_input_embeddings().weight.norm(dim=-1).mean()
            )
        current_norm = hidden_state.norm(dim=-1, keepdim=True)
        return hidden_state * (embedding_norm / current_norm)

    def loss(
        self: UninterruptedLanguageModelProtocol,
        last_hidden_state: torch.Tensor,
        labels: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        if self.config.different_lm_head_per_step and step_idx > 0:
            assert self.lm_heads is not None
            lm_head = self.lm_heads[step_idx - 1]
            print(f"Using LM head {step_idx}")
        else:
            lm_head = self.model.get_output_embeddings()

        logits = lm_head(last_hidden_state)
        logits = logits[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

        return loss

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

        seq_len = sequence.shape[1]
        num_steps = min(seq_len - 1, self.config.uninterrupted_recurrence_depth)

        last_hidden_states = None

        for i in range(num_steps):
            outputs = self._forward(self.model, sequence, attention_mask, labels)

            last_hidden_states = outputs.hidden_states[-1]

            if not self.config.train_to_backtrack or i < num_steps - 1:

                if self.config.finetune_mode != FinetuneMode.FROZEN:
                    loss = self.loss(last_hidden_states, labels, i)
                    prediction_loss = loss * self.config.loss_discount_factor**i

                    self.log(
                        f"{mode}_prediction_loss_step_{i}",
                        prediction_loss,
                        sync_dist=True,
                        batch_size=self.data_config.batch_size,
                    )

                    all_prediction_losses.append(prediction_loss)

                if self.config.uninterrupted_mode == UninterruptedMode.GMM:
                    gmm_loss = (
                        self.uninterrupted_adapter.loss(
                            last_hidden_states[:, :-1],
                            sequence[:, 1:],
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

            if i < num_steps - 1:
                next_states = (
                    # either we use the GMM to sample the next hidden state
                    # (the GMM models the distribution of the next token's HS)
                    self.uninterrupted_adapter.reparameterized_sample(
                        last_hidden_states
                    )
                    if self.config.uninterrupted_mode == UninterruptedMode.GMM
                    # or we normalize the hidden state to be closer to the embedding
                    # manifold (the embedding is the next token's HS)
                    # since Transformers end increasing the norm of the hidden states
                    # with each layer, we normalize the hidden state to
                    # have the same norm of the embedding (otherwise
                    # it would be OOD)
                    else self.normalize_hidden_state(last_hidden_states)
                )

                # we always prepend the first input_embedding again
                # that way, the sequence is always the same length
                # and we do not have to change the labels.
                # Also, the model then always has tokens at varying
                # stages of "thought" in context and hence does not get
                # confused during inference when it e.g. sees one real token
                # followed by four thought tokens and so on.
                sequence = torch.cat(
                    [
                        input_embeddings[:, :1],
                        next_states[:, :-1],
                    ],
                    dim=1,
                )

        main_prediction_loss = 0
        if self.config.train_to_backtrack:
            assert last_hidden_states is not None
            assert len(all_prediction_losses) == num_steps - 1

            if self.config.uninterrupted_mode == UninterruptedMode.GMM:
                assert len(all_hidden_state_losses) == num_steps - 1

            # Only take all the hidden states that have been processed
            # from the very beginning. E.g. for num_steps = 5, we have
            # prepended the first input_embedding after every step except
            # the last one, so 4 times. Therefore we remove the first 4
            # hidden states.
            last_hidden_states = last_hidden_states[:, num_steps - 1 :]

            assert (
                last_hidden_states.shape[1] == input_embeddings.shape[1] - num_steps + 1
            )

            # we now have num_steps - 1 fewer hidden states
            # and therefore also cut off the labels at the end
            labels_truncated = labels[:, : last_hidden_states.shape[1]]

            assert labels_truncated.shape[1] == last_hidden_states.shape[1]

            main_prediction_loss = self.loss(
                last_hidden_states, labels_truncated, num_steps - 1
            )

            self.log(
                f"{mode}_main_prediction_loss",
                main_prediction_loss,
                sync_dist=True,
                batch_size=self.data_config.batch_size,
            )

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

        total_intermediate_loss = (
            self.config.recurrent_prediction_weight * avg_prediction_loss
            + self.config.recurrent_hidden_state_weight * avg_hidden_loss
        )

        total_loss = (
            main_prediction_loss
            + self.config.uninterrupted_intermediate_supervision_loss_weight
            * total_intermediate_loss
        )

        return total_loss
