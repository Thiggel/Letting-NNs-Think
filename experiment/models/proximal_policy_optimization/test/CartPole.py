import gym
import torch
from torch import nn

from .. import ProximalPolicyOptimization, PPOConfig, PPOState

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

config = PPOConfig(
    state_dim=state_dim,
    epsilon=0.2,
    memory_capacity=512,
    batch_size=512,
    discount_factor=0.99,
    gae_lambda=0.95,
    entropy_beta=0.0,
    lr=1e-3,
    num_steps_per_update=5,
)

agent = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim),
    nn.Softmax(dim=-1),
)

ppo = ProximalPolicyOptimization(config, agent)

# Training loop
num_episodes = 500
max_timesteps = 200

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    episode_reward = 0

    ppo_states: list[PPOState] = []

    for t in range(max_timesteps):
        # Select action using PPO agent's policy
        ppo_state = ppo.act(state)
        action = ppo_state.action.item()

        # Execute action in environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Store transition in memory
        ppo_state.reward = torch.tensor([reward], dtype=torch.float32)
        ppo_state.next_state = next_state

        ppo_states.append(ppo_state)

        # Update state and accumulate reward
        state = next_state
        episode_reward += reward

        # Train PPO after every specified number of steps
        ppo.train()

        # Break if the episode is done
        if done:
            print(f"Episode {episode + 1} finished with reward {episode_reward}")
            break

        ppo.update_plot(reward=episode_reward)

    ppo.calculate_reward_and_store(ppo_states)

env.close()
