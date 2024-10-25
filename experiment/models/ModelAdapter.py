from transformers import AutoModelForCausalLM
from torch import nn

from experiment.layers import (
    RecurrentTransformerLayer,
    MambaTransformerLayer,
    GatedGemmaDecoderLayer,
    SequentialTransformerLayer,
)
from experiment.configs import ModelConfig


class ModelAdapter:
    """Handles model initialization and modification"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._initialize_model()
        self._configure_model()

    def _initialize_model(self) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, attn_implementation="eager"
        )
        model.use_cache = False
        model.train()
        return model

    def _configure_model(self):
        self._setup_finetuning()
        if self.config.make_layers_recurrent is not None:
            self._add_recurrence()

    def _setup_finetuning(self):
        """Configure which layers should be trainable"""
        if self.config.finetune_layers != "all":
            for param in self.model.parameters():
                param.requires_grad = False

            if self.config.finetune_layers is not None:
                for i in self.config.finetune_layers:
                    for param in self.model.model.layers[i].parameters():
                        param.requires_grad = True

    def _add_recurrence(self):
        """Add recurrent layers to the model"""
        start, end = self._get_recurrent_layer_range()
        layers = self.model.model.layers[start:end]

        if self.config.recurrent_mode == "mamba":
            recurrent_layer = self._create_mamba_layer(len(layers))
        elif self.config.use_gating:
            recurrent_layer = self._create_gated_layer(layers)
        else:
            recurrent_layer = SequentialTransformerLayer(*layers)

        self.model.model.layers[start] = RecurrentTransformerLayer(
            recurrent_layer,
            config=self.config,
            hidden_size=self.model.config.hidden_size,
        )

        # Remove the original layers that were made recurrent
        for i in range(start + 1, end):
            self.model.model.layers.pop(i)

        self.recurrent_layer_idx = start

    def _get_recurrent_layer_range(self) -> tuple[int, int]:
        if self.config.make_layers_recurrent is None:
            return 0, 0

        if isinstance(self.config.make_layers_recurrent, int):
            start = self.config.make_layers_recurrent
            return start, start + 1

        return self.config.make_layers_recurrent.value

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
        return SequentialTransformerLayer(*new_layers)
