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
from ..gating.TransformerGating import TransformerGating
from ..gating.GatingHooks import GatingHooks


class ModelAdapter(
    HasLayers,
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

        if self.config.finetune_mode == FinetuneMode.FULL:
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
            config.vocab_size = self.tokenizer.vocab_size
            model = AutoModelForCausalLM.from_config(
                config, attn_implementation="eager"
            )

        model.use_cache = False
        # model.gradient_checkpointing_enable()
        model.train()

        model = self._remove_layers(model)

        if self.config.untie_embedding_and_softmax:
            self._untie_embedding_and_softmax(model)

        if self.config.use_gating:
            # Convert to TransformerLens model
            hooked_model = HookedTransformer.from_pretrained(model)

            # Initialize gating
            gating = TransformerGating(hooked_model, self.config)

            # Add hooks
            hooks = GatingHooks.add_hooks(hooked_model, gating)
            hooked_model.add_hooks(hooks)

            # Store gating instance for loss computation
            model.gating = gating

            model = hooked_model

        model = self._get_peft_model(model)

        return model

    def _get_removed_layers(self, model: AutoModel) -> list[tuple[int, int]]:
        return self._get_all_layers(model, self.config.remove_layers)
