from pydantic import BaseModel, Field, ConfigDict
import torch


class PPOMemoryBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: torch.Tensor = Field(..., description="The states of the environment.")
    actions: torch.Tensor = Field(
        ..., description="The actions taken in the environment."
    )
    action_log_probs: torch.Tensor = Field(
        ..., description="The log probabilities of the actions taken."
    )
    entropies: torch.Tensor = Field(
        ..., description="The entropies of the probability distributions."
    )
    rewards: torch.Tensor = Field(
        ..., description="The rewards received from the environment."
    )
    next_states: torch.Tensor = Field(
        ..., description="The next states of the environment."
    )
