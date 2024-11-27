from typing import Protocol
import torch
from torch import nn
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM


class UninterruptedLanguageModelAdapterProtocol(Protocol):
    model: GemmaForCausalLM
    device: torch.device

    def get_decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def set_decoder_layers(
        self, model: nn.Module, layers: nn.ModuleList
    ) -> nn.Module: ...


class UninterruptedLanguageModelAdapter:
    def _untie_embedding_and_softmax(
        self: UninterruptedLanguageModelAdapterProtocol, model: GemmaForCausalLM
    ) -> None:
        new_lm_head = nn.Linear(
            model.config.hidden_size, model.config.vocab_size, bias=False
        ).to(model.device)
        new_lm_head.weight.data = model.get_output_embeddings().weight.clone().detach()
        new_lm_head.weight.requires_grad = True
        model.set_output_embeddings(new_lm_head)
        model.config.tie_word_embeddings = False

        new_embeddings = nn.Embedding(
            model.config.vocab_size, model.config.hidden_size
        ).to(model.device)
        new_embeddings.weight.data = (
            model.get_input_embeddings().weight.clone().detach()
        )
        new_embeddings.weight.requires_grad = True
        model.set_input_embeddings(new_embeddings)

    def _unfreeze_lm_head(
        self: UninterruptedLanguageModelAdapterProtocol, model: GemmaForCausalLM
    ) -> None:
        """Unfreeze the LM head parameters after LoRA wrapping"""
        # First find the actual lm_head - need to check both possible locations
        lm_head = model.get_output_embeddings()

        # Unfreeze all parameters in the lm_head
        for param in lm_head.parameters():
            param.requires_grad = True

        # Verify unfreezing worked
        print(
            "LM head requires grad:", all(p.requires_grad for p in lm_head.parameters())
        )
        print("LM head parameters:", sum(p.numel() for p in lm_head.parameters()))

    def _unfreeze_last_layer(
        self: UninterruptedLanguageModelAdapterProtocol, model: GemmaForCausalLM
    ) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.layers[-1].parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
