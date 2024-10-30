from pydantic import BaseModel, Field
import torch


class PPOMemory(BaseModel):
    state: torch.Tensor = Field(
        ...,
        description="The state of the environment. E.g. the hidden state at a certain step of the LM.",
    )
    action: torch.Tensor = Field(
        ..., description="The action taken in the environment. E.g. exit or continue."
    )
    entropy: torch.Tensor = Field(
        ..., description="The entropy of the probability distribution."
    )
    reward: torch.Tensor = Field(
        ..., description="The reward received from the environment."
    )
    next_state: torch.Tensor = Field(
        ..., description="The next state of the environment."
    )
