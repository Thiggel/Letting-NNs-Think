from pydantic import BaseModel, Field
import torch


class PPOMemoryBatch(BaseModel):
    states: torch.Tensor = Field(..., description="The states of the environment.")
    actions: torch.Tensor = Field(
        ..., description="The actions taken in the environment."
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
