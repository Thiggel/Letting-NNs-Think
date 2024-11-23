from transformers import AutoModelForCausalLM
import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from experiment.layers import (
    DynamicVeraLayer,
    MambaTransformerLayer,
    GatedGemmaDecoderLayer,
    SequentialTransformerLayer,
)
from experiment.layers.mixture_of_depths import MoDLayer
from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig


class ModelAdapter:
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
        self._configure_model()

    def _initialize_model(self) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, attn_implementation="eager"
        )
        model.use_cache = False
        model.train()

        if self.config.use_gating:
            model = self._add_gating(model)

        if self.config.use_mod:
            model = self._add_mod(model)

        if self.config.untie_embedding_and_softmax:
            self._untie_embedding_and_softmax(model)

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

        if self.config.make_uninterrupted:
            model.gradient_checkpointing_enable()

        return model

    def _untie_embedding_and_softmax(self, model: AutoModelForCausalLM) -> None:
        new_lm_head = nn.Linear(
            model.config.hidden_size, model.config.vocab_size, bias=False
        )
        new_lm_head.weight.data = model.get_input_embeddings().weight.clone().detach()
        model.lm_head = new_lm_head

    def _unfreeze_lm_head(self, model: AutoModelForCausalLM) -> None:
        for param in model.base_model.model.lm_head.parameters():
            param.requires_grad = True

    def _unfreeze_last_layer(self, model: AutoModelForCausalLM) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.layers[-1].parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True

    def _configure_model(self):
        if self.config.make_layers_recurrent is not None:
            self._add_recurrence()

    def _add_mod(self, model: nn.Module):
        model.model.layers = nn.ModuleList(
            [
                MoDLayer(
                    layer,
                    model,
                    self.config.mod_capacity,
                    self.config.mod_router_hidden_dim,
                    self.config.mod_z_loss_weight,
                    self.config.mod_capacity_loss_weight,
                    reset_mod_loss=(i == 0),
                )
                for i, layer in enumerate(model.model.layers)
            ]
        )

        return model

    def _add_gating(self, model: nn.Module):
        start, end = self._get_recurrent_layer_range(model)
        layers = model.model.layers

        for idx in range(start, end):
            layer = layers[idx]
            new_layer = GatedGemmaDecoderLayer(model.config, idx)
            new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
            new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            layers[idx] = new_layer

        model.model.layers = layers

        return model

    def _add_recurrence(self):
        """Add recurrent layers to the model"""
        start, end = self._get_recurrent_layer_range(self.model.base_model.model)
        layers = self.model.base_model.model.model.layers[start:end]

        if self.config.recurrent_mode == "mamba":
            recurrent_layer = self._create_mamba_layer(len(layers))
        else:
            recurrent_layer = SequentialTransformerLayer(*layers)

        if self.config.use_dynamic_vera:
            recurrent_layer = DynamicVeraLayer(
                recurrent_layer,
                self.model.config.hidden_size,
                self.config.vera_r,
                self.device,
            )

        self.model.base_model.model.model.layers[start] = RecurrentTransformerLayer(
            recurrent_layer,
            config=self.config,
            hidden_size=self.model.config.hidden_size,
        )

        # Remove the original layers that were made recurrent
        for i in range(start + 1, end):
            self.model.base_model.model.model.layers.pop(i)

        self.recurrent_layer_idx = start

    def _get_recurrent_layer_range(self, model: nn.Module) -> tuple[int, int]:
        if self.config.make_layers_recurrent is None:
            return 0, 0
        if ":" in self.config.make_layers_recurrent:
            start, end = map(int, self.config.make_layers_recurrent.split(":"))
        else:
            start = int(self.config.make_layers_recurrent)
            end = start + 1

        if start < 0:
            start = len(model.model.layers) + start
        if end < 0 or end == 0:
            end = len(model.model.layers) + end

        return start, end

    def _create_mamba_layer(self, num_layers: int) -> SequentialTransformerLayer:
        return SequentialTransformerLayer(
            *[
                MambaTransformerLayer(
                    self.model.config.hidden_size,
                    self.model.config.num_attention_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def _create_gated_layer(self, layers: list) -> SequentialTransformerLayer:
        new_layers = []
        for idx, layer in enumerate(layers):
            print(layer)
            new_layer = GatedGemmaDecoderLayer(self.model.config, idx)
            new_layer = get_peft_model(new_layer, self.lora_config)
            print(new_layer)
            exit()

            new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
            new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            new_layers.append(new_layer)

        return SequentialTransformerLayer(*new_layers)
