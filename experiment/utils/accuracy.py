from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def accuracy(
    outputs: CausalLMOutputWithCrossAttentions,
    labels: Tensor,
) -> float:
    predictions = outputs.logits.argmax(dim=-1)
    mask = labels != -100
    correct = ((predictions == labels) & mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0

    return accuracy
