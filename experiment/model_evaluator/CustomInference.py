import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from lm_eval.models.utils import MultiTokenEOSCriteria


from experiment.models import DefaultLightningModule


class UninterruptedTransformer(nn.Module):
    def __init__(self, model):
        super().__init__()

        print("HEYYYYY")

        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return self.model.forward(input_ids)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)

        return self

    def normalize_hidden_state(self, hidden_state, embedding_norm=None):
        """Project hidden state closer to embedding manifold"""
        if embedding_norm is None:
            embedding_norm = (
                self.model.get_input_embeddings().weight.norm(dim=-1).mean()
            )
        current_norm = hidden_state.norm(dim=-1, keepdim=True)
        return hidden_state * (embedding_norm / current_norm)

    def get_probs(self, hidden_states):
        lm_head = self.model.get_output_embeddings()
        logits = lm_head(hidden_states)

        if hasattr(self.model.config, "final_logit_softcapping"):
            softcap = self.model.config.final_logit_softcapping
            if softcap is not None:
                logits = softcap * F.tanh(logits / softcap)

        probs = F.softmax(logits, dim=-1)

        return probs

    def get_top_probs(self, hidden_states):
        probs = self.get_probs(hidden_states)

        top_probs, top_indices = torch.topk(probs[0, -1], 5)

        return top_probs, top_indices

    def get_next_token_id(self, hidden_states):
        _, top_indices = self.get_top_probs(hidden_states)

        next_token_id = top_indices[0].item()

        return torch.tensor([next_token_id], device=hidden_states.device).unsqueeze(0)

    def mix_with_embeddings(self, hidden_state, token_id, alpha=0.8):
        token_embedding = self.model.get_input_embeddings()(token_id)
        return alpha * hidden_state + (1 - alpha) * token_embedding

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens, stopping_criteria=None):
        hidden_states = self.model.get_input_embeddings()(input_ids)

        for token_idx in range(max_new_tokens):
            output = self.model.forward(
                inputs_embeds=hidden_states, return_dict=True, output_hidden_states=True
            )

            last_token = output.hidden_states[-1][:, -1:, :]
            next_token_id = self.get_next_token_id(last_token)

            normalizer = torch.tensor(
                last_token.size(-1) ** 0.5, dtype=last_token.dtype
            )
            last_token = last_token / normalizer

            last_token = self.normalize_hidden_state(last_token)

            last_token = self.mix_with_embeddings(last_token, next_token_id, alpha=0.2)

            hidden_states = torch.cat([hidden_states, last_token], dim=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if stopping_criteria:
                if stopping_criteria(input_ids, None).all():
                    break

        return input_ids


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
        embed_tokens = self.model.model.get_input_embeddings()
        lm_head = self.model.model.get_output_embeddings()

        self.model.model.set_input_embeddings(nn.Identity())
        self.model.model.set_output_embeddings(nn.Identity())

        output = self.model(*args, **kwargs).logits

        self.model.model.set_input_embeddings(embed_tokens)
        self.model.model.set_output_embeddings(lm_head)

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

        embed_tokens = self.model.model.get_input_embeddings()(input_ids)
        first_hidden_states = embed_tokens
        last_hidden_states = None

        for _ in range(max_length):
            hidden_states = self.forward_only_transformer_layers(first_hidden_states)

            last_hidden_states = hidden_states[:, -1:, :]
            first_hidden_states = torch.cat(
                (first_hidden_states, last_hidden_states), dim=1
            )

            logits = self.model.model.get_output_embeddings()(last_hidden_states)
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
