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
from experiment.models.gating import ModelGating


class ModelAdapter(HasLayers):
    """Handles model initialization and modification with LoRA and gating support"""

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

    def _wrap_with_gating(self, model: PreTrainedModel) -> PreTrainedModel:
        """Add gating wrappers to model components"""
        if not self.config.use_gating:
            return model

        # Get model dimensions
        d_model = (
            model.config.hidden_size
            if hasattr(model, "config")
            else self._infer_hidden_size(model)
        )

        # Create main gating module
        gating = ModelGating(self.config, d_model)

        # Wrap attention and MLP modules
        layers = self.get_decoder_layers(model)
        for i, layer in enumerate(layers):
            # Handle different model architectures
            if hasattr(layer, "self_attn"):
                # Handle attention
                if self.config.gate_attention:
                    layer.self_attn = gating.wrap_module(
                        f"attn_{i}", layer.self_attn, parent=layer, layer_idx=i
                    )
                # Handle MLP
                if self.config.gate_mlp and hasattr(layer, "mlp"):
                    layer.mlp = gating.wrap_module(
                        f"mlp_{i}", layer.mlp, parent=layer, layer_idx=i
                    )

            elif hasattr(layer, "attn"):
                # Handle attention
                if self.config.gate_attention:
                    layer.attn = gating.wrap_module(
                        f"attn_{i}", layer.attn, parent=layer, layer_idx=i
                    )
                # Handle MLP/FF
                if self.config.gate_mlp and hasattr(layer, "ff"):
                    layer.ff = gating.wrap_module(
                        f"mlp_{i}", layer.ff, parent=layer, layer_idx=i
                    )

        # Add gating module to model as a module to ensure proper registration
        model.add_module("gating", gating)
        return model

    def _infer_hidden_size(self, model: PreTrainedModel) -> int:
        """Infer hidden size from model parameters"""
        # Try to find a parameter that would indicate the hidden size
        for param in model.parameters():
            if len(param.shape) >= 2:
                return param.shape[-1]
        raise ValueError("Could not infer hidden size from model parameters")

    def _initialize_model(self) -> PreTrainedModel:
        """Initialize the model with appropriate configuration"""
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
        model.train()

        model = self._remove_layers(model)

        # First set requires_grad=False for all parameters if not in FULL mode
        if self.config.finetune_mode != FinetuneMode.FULL:
            for param in model.parameters():
                param.requires_grad = False

        # Add gating if needed - its parameters will have requires_grad=True by default
        if self.config.use_gating:
            model = self._wrap_with_gating(model)

        # Apply LoRA if needed
        if self.config.finetune_mode == FinetuneMode.LORA:
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()

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

        if self.config.finetune_mode == FinetuneMode.FULL:
            for param in model.parameters():
                param.requires_grad = True

        return model

    def _init_embeddings(self):
        std = 1.0 / math.sqrt(self.model.config.hidden_size)
        nn.init.normal_(self.model.get_input_embeddings().weight, std=std)
        nn.init.normal_(self.model.get_output_embeddings().weight, std=std)

    def _get_removed_layers(self, model: AutoModel) -> list[int]:
        return self._get_all_layers(model, self.config.remove_layers)
