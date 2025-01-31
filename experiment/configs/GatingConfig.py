from pydantic import Field
from enum import Enum


class GatingType(str, Enum):
    PER_LAYER = "per_layer"  # Separate gates for each layer
    SHARED = "shared"  # Shared gates across all layers


class GatingConfig:
    """Configuration for gating mechanism"""

    use_gating: bool = Field(False, description="Whether to use gating")
    gating_type: GatingType = Field("per_layer", description="Type of gating to use")
    gate_init_value: float = Field(2.2, description="Initial gate bias value")
    gate_init_std: float = Field(
        0.01, description="Initial gate weight standard deviation"
    )

    entropy_loss_weight: float = Field(
        0.01, description="Weight for entropy maximization loss"
    )
    sparsity_loss_weight: float = Field(0.01, description="Weight for sparsity loss")

    gate_attention: bool = Field(True, description="Whether to gate attention outputs")
    gate_mlp: bool = Field(True, description="Whether to gate MLP outputs")
