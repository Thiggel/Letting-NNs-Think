import torch
from torch import nn
from transformers import PreTrainedTokenizer
from lm_eval.models.utils import MultiTokenEOSCriteria

from experiment.models import DefaultLightningModule, ModelAdapter


class CustomInference:
    def __init__(
        self,
        model: DefaultLightningModule,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward_only_transformer_layers(self, *args, **kwargs):
        embed_tokens = self.model.model.model.embed_tokens
        lm_head = self.model.model.lm_head

        self.model.model.model.embed_tokens = nn.Identity()
        self.model.model.lm_head = nn.Identity()

        output = self.model(*args, **kwargs).logits

        self.model.model.model.embed_tokens = embed_tokens
        self.model.model.lm_head = lm_head

        return output

    @property
    def config(self):
        return self.model.config

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)

        return self

    def eval(self):
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        stopping_criteria: list[MultiTokenEOSCriteria] = None,
        **kwargs,
    ) -> list[str]:
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        import torch.nn.functional as F

        embed_tokens = self.model.model.model.embed_tokens(input_ids)
        first_hidden_states = embed_tokens
        last_hidden_states = None

        for _ in range(max_length):
            embed_tokens = self.model.model.model.embed_tokens(input_ids)
            first_hidden_states = embed_tokens

            hidden_states = self.forward_only_transformer_layers(first_hidden_states)

            if last_hidden_states is not None:
                first_hidden_states[:, -1:, :] = last_hidden_states

            hidden_states2 = self.forward_only_transformer_layers(first_hidden_states)
            print(F.mse_loss(hidden_states[:, -1, :], hidden_states2[:, -1, :]))

            last_hidden_states = hidden_states[:, -1:, :]
            first_hidden_states = torch.cat(
                (first_hidden_states, last_hidden_states), dim=1
            )

            logits = self.model.model.lm_head(last_hidden_states)
            next_token_ids = torch.argmax(logits, dim=-1)

            input_ids = torch.cat((input_ids, next_token_ids), dim=1)

            if stopping_criteria:
                if stopping_criteria(input_ids, None).all():
                    break

        decoded_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
        ]

        print(decoded_texts)
        exit()
        return decoded_texts
