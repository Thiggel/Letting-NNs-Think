import torch
import torch.nn as nn
import torch.nn.functional as F
import types

from experiment.configs.UninterruptedTransformerConfig import UninterruptedMode


class UninterruptedLanguageModelInference(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        alpha,
        use_adapter=False,
        beam_width=5,
        lookahead_length=5,
        temperature=1.0,
    ):
        super().__init__()
        self.model = model.model
        self.uninterrupted_recurrence_depth = (
            model.config.uninterrupted_recurrence_depth
        )
        self.train_to_backtrack = model.config.train_to_backtrack
        self.beam_width = beam_width
        self.lookahead_length = lookahead_length
        self.temperature = temperature

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
        self.use_continuous_generation = (
            self.config.uninterrupted_mode != UninterruptedMode.INTERRUPTED
        )

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

    def compute_sequence_likelihood(self, input_embeds, initial_prob=1.0):
        """Compute likelihood of a sequence through continuous generation."""
        overall_prob = initial_prob
        cont_inputs_embeds = input_embeds.clone()

        for _ in range(self.lookahead_length):
            output = self.old_forward(
                inputs_embeds=cont_inputs_embeds,
                return_dict=True,
                output_hidden_states=True,
            )

            last_hidden_state = output.hidden_states[-1][:, -1:, :]
            if self.use_continuous_generation:
                last_hidden_state = self.normalize_hidden_state(last_hidden_state)
                current_embedding = cont_inputs_embeds[:, -1:, :]
                next_token_id = self.get_next_token_id(output.logits)
                next_embedding = self.model.get_input_embeddings()(
                    next_token_id
                ).unsqueeze(1)
                last_hidden_state = (
                    last_hidden_state - current_embedding + next_embedding
                )
            else:
                next_token_id = self.get_next_token_id(output.logits)
                last_hidden_state = self.model.get_input_embeddings()(
                    next_token_id
                ).unsqueeze(1)

            cont_inputs_embeds = torch.cat(
                [cont_inputs_embeds, last_hidden_state], dim=1
            )
            probs = self.get_probs(output.logits)
            token_prob = probs[0, -1, next_token_id]
            overall_prob *= token_prob

        return overall_prob, cont_inputs_embeds

    def get_beam_logits(self, hidden_states):
        """Get logits for beam search with continuous generation."""
        logits = torch.zeros(hidden_states.size(0), self.model.config.vocab_size).to(
            self.device
        )
        top_logits, top_indices = torch.topk(
            self.get_probs(hidden_states), self.beam_width, dim=-1
        )

        for idx, token_id in enumerate(top_indices[0, -1]):
            initial_prob = top_logits[0, -1, idx]
            next_embedding = (
                self.model.get_input_embeddings()(token_id).unsqueeze(0).unsqueeze(1)
            )
            cont_inputs_embeds = torch.cat([self.inputs_embeds, next_embedding], dim=1)

            sequence_prob, _ = self.compute_sequence_likelihood(
                cont_inputs_embeds, initial_prob
            )
            logits[0, token_id] = sequence_prob

        return logits

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True

        if self.inputs_embeds is None:
            self.embed_input_ids(kwargs["input_ids"])

        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = self.inputs_embeds

        output = self.old_forward(**kwargs)
        last_hidden_state = output.hidden_states[-1][:, -1:, :]

        if self.lm_heads is not None and self.step_idx > 0:
            logits = self.lm_heads[self.step_idx - 1](last_hidden_state)
            output["logits"] = logits
        else:
            # Replace logits with beam search logits
            beam_logits = self.get_beam_logits(output.logits)
            output["logits"] = beam_logits.unsqueeze(1)

        if self.use_adapter:
            predicted_next_embedding = self.uninterrupted_adapter.sample(
                last_hidden_state, temperature=self.temperature
            )
        else:
            predicted_next_embedding = self.normalize_hidden_state(last_hidden_state)

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
            if self.use_continuous_generation:
                self.inputs_embeds = predicted_next_embedding
            else:
                next_token_id = self.get_next_token_id(output.logits)
                self.inputs_embeds = self.model.get_input_embeddings()(
                    next_token_id
                ).unsqueeze(1)
            self.step_idx += 1

        return output

    def embed_input_ids(self, input_ids):
        self.inputs_embeds = self.model.get_input_embeddings()(input_ids)

    def normalize_hidden_state(self, hidden_state, embedding_norm=None):
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
        return F.softmax(logits / self.temperature, dim=-1)

    def get_next_token_id(self, logits):
        probs = self.get_probs(logits)
        next_token_ids = torch.multinomial(probs[:, -1, :], num_samples=1)
        return next_token_ids.squeeze()

    def generate(self, *args, **kwargs):
        if self.train_to_backtrack:
            if "max_length" in kwargs:
                kwargs["max_length"] *= self.uninterrupted_recurrence_depth
            if "max_new_tokens" in kwargs:
                kwargs["max_new_tokens"] *= self.uninterrupted_recurrence_depth

        output = self.model.generate(*args, **kwargs)

        if self.train_to_backtrack:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Input IDs must be provided to calculate prompt length."
                )
            prompt_length = input_ids.shape[1]
            output = self.filter_out_thought_tokens(output, prompt_length)

        print("Generated:", self.tokenizer.decode(output[0], skip_special_tokens=True))
        self.reset_inputs_embeds()
        return output

    def filter_out_thought_tokens(self, output_ids, prompt_length):
        prompt = output_ids[:, :prompt_length]
        generated_tokens = output_ids[:, prompt_length:]
        filtered_generated_tokens = generated_tokens[
            :, :: self.uninterrupted_recurrence_depth
        ]
        return torch.cat([prompt, filtered_generated_tokens], dim=-1)
