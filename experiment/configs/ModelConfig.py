from enum import Enum
from pydantic import BaseModel, Field

from .RecursiveTransformerConfig import RecursiveTransformerConfig
from .UninterruptedTransformerConfig import UninterruptedTransformerConfig
from .LoraConfig import LoraConfig
from .DynamicVeraConfig import DynamicVeraConfig
from .MixtureOfDepthsConfig import MixtureOfDepthsConfig


class FinetuneMode(str, Enum):
    FULL = "full"
    LORA = "lora"
    UNINTERRUPTED = "lastlayer_lmhead"
    UNINTERRUPTED_LORA = "lmhead_lora"


class ModelConfig(
    BaseModel,
    RecursiveTransformerConfig,
    UninterruptedTransformerConfig,
    LoraConfig,
    DynamicVeraConfig,
    MixtureOfDepthsConfig,
):
    """Configuration for model architecture and behavior"""

    model_name: str = Field("gpt2", description="The model name to be used")
    finetune_mode: FinetuneMode = Field(
        "lora", description="The finetuning mode to use"
    )
