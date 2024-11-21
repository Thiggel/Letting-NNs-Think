from pydantic import Field


class MixtureOfDepthsConfig:
    use_mod: bool = Field(
        False, description="Whether to use the Mixture-of-Depths layer"
    )
    mod_capacity: float = Field(0.125, description="The capacity of the MoD layer")
