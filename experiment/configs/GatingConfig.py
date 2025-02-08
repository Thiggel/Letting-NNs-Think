from pydantic import Field
from enum import Enum


class GatingType(str, Enum):
    PER_LAYER = "per_layer"  # Separate gates for each layer
    SHARED = "shared"  # Shared gates across all layers


class GatingConfig:
    """Configuration for gating mechanism"""

    use_gating: bool = Field(False, description="Whether to use gating")
    actually_gate: bool = Field(
        True,
        description="Whether to just compute the gate value or actually gate the output",
    )
    skip_modules: bool = Field(
        False,
        description="Whether to, based ona gate's mean of values, skip attention and mlp modules if the contribute very little to the residual stream",
    )
    randomly_skip: bool = Field(
        False, description="Whether to randomly skip modules as a baseline"
    )
    percent_randomly_skip: float = Field(
        0.0,
        description="Percentage of modules to randomly skip if randomly_skip is True",
    )
    skip_threshold: float = Field(0.5, description="Threshold for skipping modules")
    always_skip_layer: int = Field(
        -1,
        description="If set, always skip this layer (0-indexed) regardless of gate value",
    )
    skip_module_types: list[str] = Field(
        ["attn", "mlp"], description="Module types to skip based on gate value"
    )
    skip_entire_layer_based_on_attn: bool = Field(
        False, description="Whether to skip entire layer based on attention gate value"
    )
    only_skip_every_second_layer: bool = Field(
        False, description="Whether to only skip every second layer"
    )
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
