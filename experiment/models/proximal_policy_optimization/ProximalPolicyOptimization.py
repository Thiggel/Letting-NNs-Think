from typing import Optional, Generator
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        self.parameters = lambda: (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(additional_params_to_tune_list)
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
        )

        self.init_plot()

    def init_plot(self):
        _, self.ax = plt.subplots()
        (self.actor_loss_line,) = self.ax.plot([], [], label="Actor Loss")
        (self.value_loss_line,) = self.ax.plot([], [], label="Value Loss")
        (self.reward_line,) = self.ax.plot([], [], label="Reward")
        # self.ax.legend()
        self.actor_loss = []
        self.value_loss = []
        self.reward = []
        self.running_av_reward = []

    def calculate_reward_and_store(
        self,
        ppo_states: list[PPOState],
    ) -> None:
        states = torch.stack([s.state for s in ppo_states])
        rewards = torch.tensor([s.reward for s in ppo_states], dtype=torch.float32)

        discounted_cumulative_rewards = self.calculate_discounted_cumulative_rewards(
            rewards
        )

        with torch.no_grad():
            values = self.critic(states.to(self.device)).squeeze()

        advantages = self.advantage_estimator.estimate(rewards.to(self.device), values)
        advantages = self.normalize_advantages(advantages)

        for i, ppo_state in enumerate(ppo_states):
            ppo_state.reward = discounted_cumulative_rewards[i]
            ppo_state.advantage = advantages[i]
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
            action_log_prob=action_log_prob.detach(),
            entropy=entropy.detach(),
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
            distribution * torch.log(distribution + 1e-8), dim=1
        )

    def calculate_policy_loss(
        self, ratio: torch.Tensor, advantages: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss

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
        new_log_probs = self.actor(states)
        new_dist = Categorical(new_log_probs)
        new_action_log_probs = new_dist.log_prob(actions).squeeze()
        ratio = (new_action_log_probs - old_log_probs).exp()
        value_loss = (
            self.config.value_loss_coefficient * F.mse_loss(values, rewards).mean()
        )
        actor_loss = self.calculate_policy_loss(ratio, advantages, self.config.epsilon)
        loss = actor_loss + value_loss - self.config.entropy_beta * entropies.mean()

        # self.update_plot(actor_loss, value_loss)

        return loss

    def update_plot(
        self,
        actor_loss: Optional[torch.Tensor] = None,
        value_loss: Optional[torch.Tensor] = None,
        reward: Optional[float] = None,
    ) -> None:
        if reward is not None:
            running_av_reward = (
                sum(self.reward[-300:]) / 300 if len(self.reward) > 300 else 0
            )
            self.running_av_reward.append(running_av_reward)
            self.reward.append(reward)
            self.reward_line.set_xdata(range(len(self.running_av_reward)))
            self.reward_line.set_ydata(self.running_av_reward)

        if actor_loss is not None:
            self.actor_loss.append(actor_loss.item())
            self.actor_loss_line.set_xdata(range(len(self.actor_loss)))
            self.actor_loss_line.set_ydata(self.actor_loss)

        if value_loss is not None:
            self.value_loss.append(value_loss.item())
            self.value_loss_line.set_xdata(range(len(self.value_loss)))
            self.value_loss_line.set_ydata(self.value_loss)

        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    def normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def train_step(self):
        if len(self.memory) < self.config.batch_size:
            return

        self.optimizer.zero_grad()

        batch = self.memory.sample(self.config.batch_size)

        values = self.critic(batch.states.to(self.device)).squeeze()

        loss = self.get_loss(
            batch.states.to(self.device),
            batch.actions.to(self.device),
            batch.action_log_probs.to(self.device),
            values,
            batch.advantages.to(self.device),
            batch.entropies.to(self.device),
            batch.rewards.to(self.device),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

    def update_learning_rate(self) -> None:
        self.optimizer.param_groups[0]["lr"] *= self.config.lr_decay_rate

    def train(self):
        for _ in range(self.config.num_steps_per_update):
            self.train_step()
