import torch
import torch.nn as nn
import torch.nn.functional as F
import types


class UninterruptedTransformer(nn.Module):
    def __init__(self, model, tokenizer, alpha, use_adapter=False):
        super().__init__()
        self.model = model.model

        self.uninterrupted_recurrence_depth = (
            model.config.uninterrupted_recurrence_depth
        )

        self.lm_heads = None
        if hasattr(model, "lm_heads"):
            self.lm_heads = model.lm_heads

        if use_adapter:
            self.uninterrupted_adapter = model.uninterrupted_adapter

        self.use_adapter = use_adapter

        self.tokenizer = tokenizer
        self.alpha = alpha

        self.inputs_embeds = None
        self.step_idx = 0

    def setup(self):
        self.old_forward = self.model.forward
        self.model.forward = types.MethodType(self.forward, self.model)

    def reset(self):
        self.model.forward = self.old_forward

    def reset_inputs_embeds(self):
        self.inputs_embeds = None
        self.step_idx = 0

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        return self.model.config

    def tie_weights(self):
        self.model.tie_weights()

    def embed_input_ids(self, input_ids):
        self.inputs_embeds = self.model.get_input_embeddings()(input_ids)

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True

        if self.inputs_embeds is None:
            self.embed_input_ids(kwargs["input_ids"])

        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = self.inputs_embeds

        output = self.old_forward(**kwargs)

        # Get the last hidden state
        last_hidden_state = output.hidden_states[-1][:, -1:, :]

        if self.lm_heads is not None and self.step_idx > 0:
            logits = self.lm_heads[self.step_idx - 1](last_hidden_state)
            output["logits"] = logits

        if self.use_adapter:
            predicted_next_embedding = self.uninterrupted_adapter.sample(
                last_hidden_state, temperature=0.1
            )
        else:
            predicted_next_embedding = last_hidden_state

        if not self.use_adapter:
            predicted_next_embedding = self.normalize_hidden_state(
                predicted_next_embedding
            )

        if (
            self.uninterrupted_recurrence_depth is not None
            and self.step_idx >= self.uninterrupted_recurrence_depth - 1
        ):
            self.reset_inputs_embeds()
            next_token_id = self.get_next_token_id(output.logits)
            token_embedding = self.model.get_input_embeddings()(
                next_token_id
            ).unsqueeze(1)
            self.inputs_embeds = token_embedding

        else:
            self.inputs_embeds = predicted_next_embedding
            self.step_idx += 1

        return output

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

    def get_probs(self, logits):
        if hasattr(self.model.config, "final_logit_softcapping"):
            softcap = self.model.config.final_logit_softcapping
            if softcap is not None:
                logits = softcap * F.tanh(logits / softcap)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_top_probs(self, hidden_states):
        probs = self.get_probs(hidden_states)
        top_probs, top_indices = torch.topk(probs[:, -1, :], 5, dim=-1)

        return top_probs, top_indices

    def get_next_token_id(self, logits):
        next_token_ids = torch.argmax(logits, dim=-1)
        return next_token_ids.squeeze()

    def mix_with_embeddings(self, hidden_state, token_id, alpha=0.8):
        token_embedding = self.model.get_input_embeddings()(token_id)
        return alpha * hidden_state + (1 - alpha) * token_embedding

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        output = self.model.generate(
            *args,
            **kwargs,
        )

        print("Generated: ", self.tokenizer.decode(output[0], skip_special_tokens=True))
        self.reset_inputs_embeds()

        return output
