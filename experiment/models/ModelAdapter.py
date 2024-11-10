from transformers import AutoModelForCausalLM
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from experiment.layers import (
    MambaTransformerLayer,
    GatedGemmaDecoderLayer,
    SequentialTransformerLayer,
)
from experiment.layers.recurrent_transformer_layer import RecurrentTransformerLayer
from experiment.configs import ModelConfig


class ModelAdapter:
    """Handles model initialization and modification with LoRA support"""

    def __init__(self, config: ModelConfig):
        self.config = config

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
        if self.config.finetune_mode == "lora":
            print("Using LoRA")
            model = get_peft_model(model, self.lora_config)
        elif self.config.finetune_mode == "full":
            print("Using full finetuning")

        model.print_trainable_parameters()

        return model

    def _configure_model(self):
        if self.config.make_layers_recurrent is not None:
            self._add_recurrence()

    def _add_recurrence(self):
        """Add recurrent layers to the model"""
        start, end = self._get_recurrent_layer_range()
        layers = self.model.base_model.model.model.layers[start:end]

        if self.config.recurrent_mode == "mamba":
            recurrent_layer = self._create_mamba_layer(len(layers))
        elif self.config.use_gating:
            recurrent_layer = self._create_gated_layer(layers)
        else:
            recurrent_layer = SequentialTransformerLayer(*layers)

        self.model.base_model.model.model.layers[start] = RecurrentTransformerLayer(
            recurrent_layer,
            config=self.config,
            hidden_size=self.model.config.hidden_size,
        )

        # Remove the original layers that were made recurrent
        for i in range(start + 1, end):
            self.model.base_model.model.model.layers.pop(i)

        self.recurrent_layer_idx = start

    def _get_recurrent_layer_range(self) -> tuple[int, int]:
        if self.config.make_layers_recurrent is None:
            return 0, 0
        if ":" in self.config.make_layers_recurrent:
            start, end = map(int, self.config.make_layers_recurrent.split(":"))
            return start, end
        start = int(self.config.make_layers_recurrent)
        return start, start + 1

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
            new_layer = GatedGemmaDecoderLayer(self.model.config, idx)
            new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
            new_layer.mlp.load_state_dict(layer.mlp.state_dict())
            new_layers.append(new_layer)
            new_layer = get_peft_model(new_layer, self.lora_config)

        return SequentialTransformerLayer(*new_layers)
