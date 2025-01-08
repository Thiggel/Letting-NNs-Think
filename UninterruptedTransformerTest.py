import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

model_name = "meta-llama/Llama-3.2-1B"

# Load model and setup as before
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_model.to(device)

print(hf_model)

output = hf_model.generate(
    tokenizer.encode("My dog is ", return_tensors="pt").to(device),
    max_new_tokens=20,
    do_sample=False,
)

print(
    "\n\n",
    tokenizer.decode(
        output.tolist()[0],
        skip_special_tokens=True,
    ),
    "\n\n",
)


def create_attention_mask(x):
    attention_mask = (
        torch.triu(
            -torch.inf * torch.ones(x.size(1), x.size(1)).to(x.device),
            diagonal=1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(x.device)
    )

    return attention_mask


class HookedLayer(nn.Module):
    def __init__(
        self,
        model,
        layer_idx,
        layer,
        tokenizer,
        is_last=False,
        input_ids=None,
        input_embeds=None,
    ):
        super().__init__()

        self.model = model
        self.layer_idx = layer_idx
        self.layer = layer
        self.tokenizer = tokenizer

    def get_probs(self, x):
        lm_head = self.model.get_output_embeddings()
        logits = lm_head(x)

        if hasattr(hf_model.config, "final_logit_softcapping"):
            softcap = hf_model.config.final_logit_softcapping
            if softcap is not None:
                logits = softcap * F.tanh(logits / softcap)

        probs = F.softmax(logits, dim=-1)

        return probs

    def get_top_probs(self, x):
        probs = self.get_probs(x)

        top_probs, top_indices = torch.topk(probs[0, -1], 5)

        return top_probs, top_indices

    def print_top_predictions(self, x):
        top_probs, top_indices = self.get_top_probs(x)

        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx])
            print(f"  {token}: {prob:.5f}")

    def forward(self, x, *args, verbose=False, **kwargs):

        if self.layer_idx == 0 and verbose:
            print("first layer", x[0, -1, :])
            print(f"\nPredictions before layer {self.layer_idx}:")
            self.print_top_predictions(x)

        output = self.layer(x, *args, **kwargs)

        if self.is_last:
            last_hidden_state = output[0]

            print("Predictions after last layer:")
            self.print_top_predictions(last_hidden_state)

        return output


input_text = "My dog is "
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

input_embeds = hf_model.get_input_embeddings()(input_ids)

# for layer_idx in range(hf_model.config.num_hidden_layers):
#    hf_model.model.layers[layer_idx] = HookedLayer(
#        hf_model,
#        layer_idx,
#        hf_model.model.layers[layer_idx],
#        tokenizer,
#    )


class UninterruptedTransformer(nn.Module):
    def __init__(self, model, max_new_tokens=100):
        super().__init__()

        self.model = model

        self.max_new_tokens = max_new_tokens

    def forward(self, input_ids: torch.Tensor):
        return self.model.forward(input_ids)

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

        if hasattr(hf_model.config, "final_logit_softcapping"):
            softcap = hf_model.config.final_logit_softcapping
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

    def generate(self, input_ids: torch.Tensor):
        hidden_states = self.model.get_input_embeddings()(input_ids)

        for token_idx in range(self.max_new_tokens):
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

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return input_ids


uninterrupted_model = UninterruptedTransformer(hf_model, max_new_tokens=20)

with torch.no_grad():

    output = uninterrupted_model.generate(input_ids)
    decoded = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print("Generated:", decoded)
