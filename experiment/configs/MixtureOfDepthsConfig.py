from pydantic import Field


class MixtureOfDepthsConfig:
    use_mod: bool = Field(
        False, description="Whether to use the Mixture-of-Depths layer"
    )
    mod_capacity: float = Field(0.125, description="The capacity of the MoD layer")
    mod_router_hidden_dim: int = Field(
        256, description="The hidden dimension of the MoD router"
    )
    mod_z_loss_weight: float = Field(0.001, description="The weight of the Z-loss")
    mod_capacity_loss_weight: float = Field(
        0.001, description="The weight of the capacity loss"
    )
