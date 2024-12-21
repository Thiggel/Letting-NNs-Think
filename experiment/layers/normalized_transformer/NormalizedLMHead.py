import torch
from torch import nn


class NormalizedLMHead(nn.Module):
    def __init__(self, lm_head: nn.Linear):
        super().__init__()
        self.lm_head = lm_head
        self.logit_scaling = nn.Parameter(torch.ones(lm_head.out_features))

    @property
    def weight(self):
        return self.lm_head.weight

    def forward(self, *args, **kwargs):
        logits = self.lm_head(*args, **kwargs) * self.logit_scaling

        return logits
