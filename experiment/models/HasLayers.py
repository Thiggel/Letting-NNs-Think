from transformers import AutoModel
from torch import nn


class HasLayers:
    def get_decoder_layers(self, model: AutoModel) -> nn.ModuleList:
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            model = model.base_model.model

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Gemma2 and similar architectures
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT and similar architectures
            return model.transformer.h
        elif hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            # Models like T5 (encoder-decoder, but using decoder only)
            return model.decoder.block
        else:
            raise ValueError(
                "Unable to locate decoder layers. Inspect the model structure."
            )

    def set_decoder_layers(self, model: AutoModel, layers: nn.ModuleList):
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            model = model.base_model.model

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Gemma2 and similar architectures
            model.model.layers = layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT and similar architectures
            model.transformer.h = layers
        elif hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            # Models like T5 (encoder-decoder, but using decoder only)
            model.decoder.block = layers
        else:
            raise ValueError(
                "Unable to locate decoder layers. Inspect the model structure."
            )

        return model
