import torch
import torch.nn as nn
import torch.nn.functional as F
import types


class UninterruptedTransformer(nn.Module):
    def __init__(self, model, tokenizer, alpha):
        super().__init__()
        self.model = model.model

        self.uninterrupted_adapter = model.uninterrupted_adapter

        self.tokenizer = tokenizer
        self.alpha = alpha

        self.inputs_embeds = None

    def setup(self):
        self.old_forward = self.model.forward
        self.model.forward = types.MethodType(self.forward, self.model)

    def reset(self):
        self.model.forward = self.old_forward

    def reset_inputs_embeds(self):
        self.inputs_embeds = None

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

    def interpolate_embeddings(self, hidden_state, top_embeddings, interpolate_top_k):
        # Create exponentially decaying weights that sum to 1
        # Start with highest weight for most likely token
        weights = torch.tensor(
            [self.alpha**i for i in range(interpolate_top_k)],
            device=top_embeddings.device,
        )
        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Reshape weights for broadcasting: [batch_size, top_k, 1]
        weights = weights.view(1, -1, 1)

        # Select the top k embeddings: [batch_size, top_k, hidden_dim]
        selected_embeddings = top_embeddings[:, :interpolate_top_k, :]

        # Weighted sum of embeddings
        interpolated = (selected_embeddings * weights).sum(dim=1, keepdim=True)

        return interpolated

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

        predicted_next_embedding = self.uninterrupted_adapter.sample(
            last_hidden_state, temperature=0.1
        )

        # Get next token
        # _, top_indices = self.get_top_probs(last_hidden_state)
        # top_embeddings = self.model.get_input_embeddings()(top_indices)
        # interpolate_top_k = 3
        # interpolated = self.interpolate_embeddings(
        #    last_hidden_state, top_embeddings, interpolate_top_k
        # )
        # last_hidden_state = interpolated
        next_token_id = self.get_next_token_id(last_hidden_state)

        # Process the last token as before
        # normalizer = torch.tensor(
        #    last_hidden_state.size(-1) ** 0.5,
        #    dtype=last_hidden_state.dtype,
        #    device=last_hidden_state.device,
        # )
        # last_hidden_state = last_hidden_state / normalizer
        # last_hidden_state = self.normalize_hidden_state(last_hidden_state)

        # Mix with embeddings
        predicted_next_embedding = self.mix_with_embeddings(
            predicted_next_embedding, next_token_id, alpha=self.alpha
        )

        self.inputs_embeds = predicted_next_embedding.unsqueeze(1)

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
        top_probs, top_indices = torch.topk(probs[:, -1, :], 5, dim=-1)

        return top_probs, top_indices

    def get_next_token_id(self, hidden_states):
        probs = self.get_probs(hidden_states)
        next_token_ids = torch.argmax(probs, dim=-1)
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

        print("Generated: ", self.tokenizer.decode(output[0]))

        self.reset_inputs_embeds()

        return output
