from collections import deque
from typing import Deque
import torch

from .PPOMemory import PPOMemory
from .PPOMemoryBatch import PPOMemoryBatch


class PPOMemoryBuffer:
    def __init__(self, capacity: int):
        self.memory: Deque[PPOMemory] = deque(maxlen=capacity)

    def add(self, memory: PPOMemory):
        self.memory.append(memory)

    def sample(self, batch_size: int) -> PPOMemoryBatch:
        indices = torch.randperm(len(self.memory))[:batch_size]
        batch = [self.memory[idx] for idx in indices]

        states = torch.stack([m.state for m in batch])
        actions = torch.stack([m.action for m in batch])
        rewards = torch.stack([m.reward for m in batch])
        entropies = torch.stack([m.entropy for m in batch])
        next_states = torch.stack([m.next_state for m in batch])

        return PPOMemoryBatch(
            states=states,
            actions=actions,
            entropies=entropies,
            rewards=rewards,
            next_states=next_states,
        )
