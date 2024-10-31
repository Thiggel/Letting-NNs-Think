from pydantic import BaseModel, Field


class PPOConfig(BaseModel):
    state_dim: int = Field(..., description="The dimension of the state space")
    epsilon: float = Field(0.2, description="The clipping parameter epsilon")
    memory_capacity: int = Field(10000, description="The capacity of the memory")
    batch_size: int = Field(32, description="The batch size")
    discount_factor: float = Field(0.99, description="The discount factor gamma")
    gae_lambda: float = Field(0.95, description="The GAE lambda parameter")
    entropy_beta: float = Field(
        3e-4, description="The entropy regularization parameter"
    )
    lr: float = Field(1e-3, description="The learning rate")
    num_steps_per_update: int = Field(
        10, description="The number of training steps per update"
    )
    max_grad_norm: float = Field(0.5, description="The maximum gradient norm")
    value_loss_coefficient: float = Field(0.5, description="The value loss coefficient")
    lr_decay_rate: float = Field(0.99, description="The learning rate decay rate")
    kl_beta: float = Field(
        1.0, description="The KL divergence regularization parameter"
    )
    kl_target: float = Field(0.01, description="The target KL divergence value")
