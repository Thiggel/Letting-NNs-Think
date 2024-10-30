from typing import Optional, Generator
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from .AdvantageEstimator import AdvantageEstimator
from .GAE import GAE
from .PPOConfig import PPOConfig
from .PPOState import PPOState
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
        torch.autograd.set_detect_anomaly(True)
        self.config = config

        self.memory = PPOMemoryBuffer(config.memory_capacity)

        self.actor = actor

        self.device = next(self.actor.parameters()).device

        self.critic = nn.Sequential(
            nn.Linear(config.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.critic.to(self.device)

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

    def calculate_reward_and_store(
        self,
        ppo_states: list[PPOState],
    ) -> None:
        rewards = torch.tensor([s.reward for s in ppo_states], dtype=torch.float32)

        discounted_cumulative_rewards = self.calculate_discounted_cumulative_rewards(
            rewards
        )

        for i, ppo_state in enumerate(ppo_states):
            ppo_state.reward = discounted_cumulative_rewards[i].detach()
            ppo_state.action_log_prob = ppo_state.action_log_prob.detach()
            ppo_state.entropy = ppo_state.entropy.detach()

            self.store_transition(ppo_state)

    def store_transition(
        self,
        ppo_state: PPOState,
    ):
        self.memory.add(
            ppo_state,
        )

    def act(self, state: torch.Tensor) -> PPOState:
        probs = self.actor(state.to(self.device))
        dist = Categorical(probs)
        entropy = self.calculate_entropy(probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).squeeze()

        return PPOState(
            state=state,
            action=action,
            action_log_prob=action_log_prob,
            entropy=entropy,
            reward=None,
            next_state=None,
        )

    def calculate_discounted_cumulative_rewards(
        self, rewards: torch.Tensor
    ) -> torch.Tensor:
        discounted_cumulative_rewards = torch.zeros_like(rewards)
        discounted_cumulative_rewards[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            discounted_cumulative_rewards[t] = (
                rewards[t]
                + self.config.discount_factor * discounted_cumulative_rewards[t + 1]
            )

        return discounted_cumulative_rewards

    def calculate_entropy(self, distribution: torch.Tensor) -> torch.Tensor:
        return -self.config.entropy_beta * torch.sum(
            distribution * torch.log(distribution), dim=1
        )

    def get_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        entropies: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        new_log_probs = self.actor(states).log()
        action_mask = F.one_hot(actions, new_log_probs.shape[-1]).bool()
        new_action_log_probs = new_log_probs[action_mask]
        ratio = (new_action_log_probs - old_log_probs).exp()
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
        if len(self.memory) < self.config.batch_size:
            return

        self.optimizer.zero_grad()

        batch = self.memory.sample(self.config.batch_size)

        values = self.critic(batch.states.to(self.device)).squeeze()

        advantages = self.advantage_estimator.estimate(
            batch.rewards.to(self.device), values
        )

        loss = self.get_loss(
            batch.states.to(self.device),
            batch.actions.to(self.device),
            batch.action_log_probs.to(self.device),
            values,
            advantages,
            batch.entropies.to(self.device),
            batch.rewards.to(self.device),
        )

        loss.backward()
        self.optimizer.step()

    def train(self):
        for _ in range(self.config.num_steps_per_update):
            self.train_step()
