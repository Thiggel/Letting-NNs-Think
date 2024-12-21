import torch
from torch import nn
from torch.nn.parameter import Parameter


class NormalizedLMHead(nn.Module):
    def __init__(self, lm_head: nn.Linear):
        super().__init__()
        # Instead of wrapping the lm_head, we'll create our own parameters
        self.weight = Parameter(lm_head.weight.data.clone())
        if hasattr(lm_head, "bias") and lm_head.bias is not None:
            self.bias = Parameter(lm_head.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        self.logit_scaling = Parameter(torch.ones(lm_head.out_features))
        self.in_features = lm_head.in_features
        self.out_features = lm_head.out_features

    def forward(self, hidden_states):
        logits = nn.functional.linear(hidden_states, self.weight, self.bias)
        return logits * self.logit_scaling

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # If loading from a non-normalized model, handle the weight transfer
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            self.weight.data.copy_(state_dict[weight_key])
            if self.bias is not None and prefix + "bias" in state_dict:
                self.bias.data.copy_(state_dict[prefix + "bias"])

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
