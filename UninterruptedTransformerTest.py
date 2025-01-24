from warnings import warn
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

        output = self.layer(x, *args, **kwargs)

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
    def __init__(
        self, model, max_new_tokens=100, skip_first_half=False, tokenizer=None
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.max_new_tokens = max_new_tokens

        self.skip_first_half = skip_first_half
        self.has_already_skipped = False
        self.removed_layers = nn.ModuleList()

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
        probs = self.get_probs(hidden_states)
        top_5_probs, top_5_indices = torch.topk(probs[:, -1, :], 5, dim=-1)
        next_token_ids = top_5_indices[:, 0]

        # for i in range(5):
        #    token_id = top_5_indices[0, i].item()
        #    token = self.tokenizer.decode([token_id])
        #    print(f"Top {i+1} token = {token}: {top_5_probs[0, i]:.5f}")
        # print()

        return next_token_ids.unsqueeze(0)

    def get_removed_layers_norm(self, hidden_state):
        current_state = hidden_state
        position_ids = (
            torch.arange(0, current_state.size(1), device=current_state.device)
            .unsqueeze(0)
            .repeat(current_state.size(0), 1)
        )
        with torch.no_grad():
            for layer in self.removed_layers:
                current_state = layer(
                    current_state,
                    position_ids=position_ids,
                )[0]

        return current_state.norm(dim=-1, keepdim=True)[:, -1:, :]

    def mix_with_embeddings(self, hidden_state, token_id, alpha=0.8):
        token_embedding = self.model.get_input_embeddings()(token_id)
        return alpha * hidden_state + (1 - alpha) * token_embedding

    def generate(self, input_ids: torch.Tensor):
        hidden_states = self.model.get_input_embeddings()(input_ids)

        for token_idx in range(self.max_new_tokens):
            if token_idx == 1 and self.skip_first_half and not self.has_already_skipped:
                num_layers = len(self.model.model.layers)
                for layer_idx in range(num_layers):
                    if layer_idx < num_layers // 2:
                        print(f"Removing layer {layer_idx}")
                        self.removed_layers.append(self.model.model.layers.pop(0))

                self.has_already_skipped = True

                print(self.model)

            output = self.model.forward(
                inputs_embeds=hidden_states,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )

            last_token = output.hidden_states[-1][:, -1:, :]
            next_token_id = self.get_next_token_id(last_token)

            # last_token_normalized = self.model.model.norm(last_token)
            last_token_normalized = self.normalize_hidden_state(last_token)

            if len(self.removed_layers) > 0:
                norm = self.get_removed_layers_norm(
                    torch.cat([hidden_states, last_token_normalized], dim=1)
                )

                last_token_normalized = self.normalize_hidden_state(
                    last_token, embedding_norm=norm
                )

            last_token_mixed = self.mix_with_embeddings(
                last_token_normalized, next_token_id, alpha=1.0
            )

            hidden_states = torch.cat([hidden_states, last_token_mixed], dim=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return input_ids


uninterrupted_model = UninterruptedTransformer(
    hf_model, max_new_tokens=20, skip_first_half=False, tokenizer=tokenizer
)

with torch.no_grad():

    output = uninterrupted_model.generate(input_ids)
    decoded = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print("Generated:", decoded)
