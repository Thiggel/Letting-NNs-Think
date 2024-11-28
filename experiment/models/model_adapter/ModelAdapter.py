from transformers import AutoModelForCausalLM, PreTrainedModel
import torch
from peft import get_peft_model, LoraConfig, TaskType
from experiment.configs import ModelConfig

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

    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
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

    def _get_peft_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if self.config.finetune_mode == "lora":
            print("Using LoRA")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()
        elif self.config.finetune_mode == "full":
            print("Using full finetuning")
        elif self.config.finetune_mode == "lastlayer_lmhead":
            self._unfreeze_last_layer(model)
        elif self.config.finetune_mode == "lmhead_lora":
            model = get_peft_model(model, self.lora_config)
            self._unfreeze_lm_head(model)
            model.print_trainable_parameters()

        return model

    def _initialize_model(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, attn_implementation="eager"
        )
        model.use_cache = False
        model.train()

        if self.config.use_gating:
            model = self._add_gating(model)

        if self.config.enable_normalization:
            model = self._add_normalization(model)

        if self.config.use_mod:
            model = self._add_mod(model)

        model = self._get_peft_model(model)

        if self.config.untie_embedding_and_softmax:
            self._untie_embedding_and_softmax(model)

        if self.config.make_uninterrupted:
            model.gradient_checkpointing_enable()

        if self.config.make_layers_recurrent is not None:
            model = self._add_recurrence(model)

        return model
