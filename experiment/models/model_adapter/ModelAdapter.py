from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoConfig,
    PreTrainedTokenizer,
)
import math
import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from experiment.configs import ModelConfig, FinetuneMode

from ..HasLayers import HasLayers
from .MoDAdapter import MoDAdapter
from .NormalizedLanguageModelAdapter import NormalizedLanguageModelAdapter
from .RecurrentLanguageModelAdapter import RecurrentLanguageModelAdapter
from .GatedLanguageModelAdapter import GatedLanguageModelAdapter
from .UninterruptedLanguageModelAdapter import UninterruptedLanguageModelAdapter


class ModelAdapter(
    HasLayers,
    MoDAdapter,
    NormalizedLanguageModelAdapter,
    RecurrentLanguageModelAdapter,
    GatedLanguageModelAdapter,
    UninterruptedLanguageModelAdapter,
):
    """Handles model initialization and modification with LoRA support"""

    def __init__(
        self, config: ModelConfig, tokenizer: PreTrainedTokenizer, device: torch.device
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        self.lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = self._initialize_model()

        if not self.config.pretrained:
            self._init_embeddings()

        if self.config.enable_normalization:
            self.normalize_weights()

    def _adjust_embedding_size(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.tokenizer.vocab_size != model.config.vocab_size:
            model.resize_token_embeddings(self.tokenizer.vocab_size)

        return model

    def _remove_layers(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.config.remove_layers is not None:
            removed_layers = self._get_removed_layers(model)
            layers = self.get_decoder_layers(model)
            layers = nn.ModuleList(
                [layer for idx, layer in enumerate(layers) if idx not in removed_layers]
            )
            model = self.set_decoder_layers(model, layers)

        return model

    def _get_peft_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.config.finetune_mode == FinetuneMode.LORA:
            print("Using LoRA")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()

            return model

        for param in model.parameters():
            param.requires_grad = False

        if self.config.finetune_mode == FinetuneMode.UNINTERRUPTED:
            self._unfreeze_last_layer(model)

        elif self.config.finetune_mode == FinetuneMode.UNINTERRUPTED_LORA:
            model = get_peft_model(model, self.lora_config)
            self._unfreeze_lm_head(model)
            model.print_trainable_parameters()

        elif self.config.finetune_mode == FinetuneMode.FULL:
            for param in model.parameters():
                param.requires_grad = True

        return model

    def _init_embeddings(self):
        std = 1.0 / math.sqrt(self.model.config.hidden_size)
        nn.init.normal_(self.model.get_input_embeddings().weight, std=std)
        nn.init.normal_(self.model.get_output_embeddings().weight, std=std)

    def _initialize_model(self) -> PreTrainedModel:
        if self.config.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, attn_implementation="eager"
            )
        else:
            config = AutoConfig.from_pretrained(self.config.model_name)
            model = AutoModelForCausalLM.from_config(
                config, attn_implementation="eager"
            )

        model.use_cache = False
        model.train()

        model = self._remove_layers(model)
        model = self._adjust_embedding_size(model)

        if self.config.untie_embedding_and_softmax:
            self._untie_embedding_and_softmax(model)

        if self.config.use_gating:
            model = self._add_gating(model)

        if self.config.enable_normalization:
            model = self._add_normalization(model)

        if self.config.use_mod:
            model = self._add_mod(model)

        model = self._get_peft_model(model)

        if self.config.make_layers_recurrent is not None:
            model = self._add_recurrence(model)

        return model

    def _get_removed_layers(self, model: AutoModel) -> list[tuple[int, int]]:
        return self._get_all_layers(model, self.config.remove_layers)
