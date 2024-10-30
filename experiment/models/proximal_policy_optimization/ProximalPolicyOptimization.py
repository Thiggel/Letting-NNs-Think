from typing import Optional, Generator
import torch
from torch import nn
import torch.nn.functional as F

from .AdvantageEstimator import AdvantageEstimator
from .GAE import GAE
from .PPOConfig import PPOConfig
from .PPOMemory import PPOMemory
from .PPOMemoryBuffer import PPOMemoryBuffer
from .AdvantageEstimator import AdvantageEstimator


class ProximalPolicyOptimization:
    def __init__(
        self,
        config: PPOConfig,
        actor: nn.Module,
        advantage_estimator: Optional[AdvantageEstimator] = None,
        additional_params_to_tune: Optional[Generator[nn.Parameter, None, None]] = None,
    ):
        self.config = config

        self.memory = PPOMemoryBuffer(config.memory_capacity)

        # we need to keep a copy of the old actor
        # for the loss calculation
        self.old_actor = actor
        self.actor = actor

        self.critic = nn.Sequential(
            nn.Linear(config.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_estimator = advantage_estimator or GAE(
            config.discount_factor, config.gae_lambda
        )

        additional_params_to_tune_list = (
            list(additional_params_to_tune)
            if additional_params_to_tune is not None
            else []
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(additional_params_to_tune_list),
            lr=self.config.lr,
        )

        # TODO: have this whole thing work on a specific GPU like cuda:2 or maybe just any bco deepspeed stage 3
        # TODO: have a function to calculate discounted cumulative rewards, entropy etc and put that into the memory

    def store_transition(self, state, action, reward, entropy, next_state):
        self.memory.add(
            PPOMemory(
                state=state,
                action=action,
                reward=reward,
                entropy=entropy,
                next_state=next_state,
            )
        )

    def get_loss(self, states, values, advantages, entropies, rewards):
        new_log_probs = self.actor(states).log()
        old_log_probs = self.old_actor(states).detach().log()
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
            * advantages
        )
        value_loss = F.mse_loss(values, rewards).mean()
        loss = (
            -torch.min(surr1, surr2).mean()
            + value_loss
            - self.config.entropy_beta * entropies.mean()
        )

        return loss

    def train_step(self):
        batch = self.memory.sample(self.config.batch_size)

        values = self.critic(batch.states)

        advantages = self.advantage_estimator.estimate(batch.rewards, values)

        loss = self.get_loss(
            batch.states, values, advantages, batch.entropies, batch.rewards
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for _ in range(self.config.num_steps_per_update):
            self.train_step()
