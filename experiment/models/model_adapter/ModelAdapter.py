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
from experiment.models.gating.GateLayer import GateLayer
from ..HasLayers import HasLayers
from experiment.models.gating import ModelGating
from experiment.models.mixture_of_depths import ModelMod


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

    def _wrap_with_gating_or_mod(self, model: PreTrainedModel) -> PreTrainedModel:
        """Add gating or MoD wrappers to model components"""
        if not (self.config.use_gating or self.config.use_mod):
            return model

        # Get model dimensions
        d_model = (
            model.config.hidden_size
            if hasattr(model, "config")
            else self._infer_hidden_size(model)
        )

        # Create main routing module (either gating or MoD)
        if self.config.use_mod:
            routing = ModelMod(self.config, d_model)
        else:
            routing = ModelGating(self.config, d_model)

        if self.config.use_mod:
            # Create MoD routing
            routing = ModelMod(self.config, d_model)

            # Get decoder layers
            layers = self.get_decoder_layers(model)

            # Wrap entire decoder layers
            for i, layer in enumerate(layers):
                wrapped_layer = routing.wrap_module(
                    f"layer_{i}",
                    layer,
                    parent=model,  # Parent is now the model itself
                    layer_idx=i,
                )
                # Replace the layer in the model's layers
                layers[i] = wrapped_layer

            # Add routing module to model
            model.add_module("mod", routing)
        else:
            # Original gating logic
            attn_gate = (
                GateLayer(d_model, self.config)
                if not self.config.one_gate_per_layer
                else None
            )
            mlp_gate = (
                GateLayer(d_model, self.config)
                if not self.config.one_gate_per_layer
                else None
            )

            layers = self.get_decoder_layers(model)
            for i, layer in enumerate(layers):
                if hasattr(layer, "self_attn"):
                    if self.config.gate_attention:
                        layer.self_attn = routing.wrap_module(
                            f"attn_{i}",
                            layer.self_attn,
                            parent=layer,
                            layer_idx=i,
                            gate=attn_gate,
                        )
                    if self.config.gate_mlp and hasattr(layer, "mlp"):
                        layer.mlp = routing.wrap_module(
                            f"mlp_{i}",
                            layer.mlp,
                            parent=layer,
                            layer_idx=i,
                            gate=mlp_gate,
                        )
                elif hasattr(layer, "attn"):
                    if self.config.gate_attention:
                        layer.attn = routing.wrap_module(
                            f"attn_{i}",
                            layer.attn,
                            parent=layer,
                            layer_idx=i,
                            gate=attn_gate,
                        )
                    if self.config.gate_mlp and hasattr(layer, "ff"):
                        layer.ff = routing.wrap_module(
                            f"mlp_{i}",
                            layer.ff,
                            parent=layer,
                            layer_idx=i,
                            gate=mlp_gate,
                        )

        # Add routing module to model
        if self.config.use_mod:
            model.add_module("mod", routing)
        else:
            model.add_module("gating", routing)

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

        # Add gating or MoD if needed
        model = self._wrap_with_gating_or_mod(model)

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
