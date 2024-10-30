from collections import deque
from typing import Deque
import torch

from .PPOState import PPOState
from .PPOMemoryBatch import PPOMemoryBatch


class PPOMemoryBuffer:
    def __init__(self, capacity: int):
        self.memory: Deque[PPOState] = deque(maxlen=capacity)

    def add(self, memory: PPOState):
        self.memory.append(memory)

    def sample(self, batch_size: int) -> PPOMemoryBatch:
        indices = torch.randperm(len(self.memory))[:batch_size]
        batch = [self.memory[idx] for idx in indices]

        states = torch.stack([m.state for m in batch])
        actions = torch.stack([m.action for m in batch])
        action_log_probs = torch.stack([m.action_log_prob for m in batch])
        rewards = torch.stack([m.reward for m in batch])
        entropies = torch.stack([m.entropy for m in batch])
        next_states = torch.stack([m.next_state for m in batch])

        return PPOMemoryBatch(
            states=states,
            actions=actions,
            action_log_probs=action_log_probs,
            entropies=entropies,
            rewards=rewards,
            next_states=next_states,
        )

    def __len__(self):
        return len(self.memory)
