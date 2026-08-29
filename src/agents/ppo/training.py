from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any

import gymnasium as gym
import torch
from torch.utils.data import DataLoader, Dataset

from src.agents.ppo.agent import PPOAgent

PPO_EPS = 0.2


@dataclass
class RolloutBatch:
    states: torch.Tensor
    next_states: torch.Tensor
    raw_actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    log_probs: torch.Tensor


class PPODataset(Dataset):
    def __init__(self, raw_data, transforms=None):
        transforms = transforms or {}

        T, N = raw_data["rewards"].shape
        dataset_size = T * N

        self.data = {k: v.reshape(dataset_size, *v.shape[2:]) for k, v in raw_data.items()}
        self.data = {k: transforms[k](v) if k in transforms else v for k, v in self.data.items()}
        self.size = next(iter(self.data.values())).shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        envs: gym.vector.VectorEnv,
        device: torch.device,
        num_steps: int = 1024,
        k_epoch: int = 4,
        batch_size: int = 32,
    ):
        self.agent = agent
        self.envs = envs
        self.best_reward_tracker = [float("-inf")]
        self.total_episodes = 0
        self.num_steps = num_steps
        self.k_epoch = k_epoch
        self.batch_size = batch_size

        self.device = device

    def train(
        self,
        num_episodes=100,
        transforms=None,
        after_episode_callback: None | Callable[[int, list[Any]], None] = None,
    ):

        for episode in count():
            if episode >= num_episodes:
                break

            trajectory, meta = self.rollout(episode)
            advantages, values = self.advantage(trajectory)

            ds = PPODataset(
                raw_data={
                    "states": trajectory.states,
                    "next_states": trajectory.next_states,
                    "raw_actions": trajectory.raw_actions,
                    "rewards": trajectory.rewards,
                    "terminated": trajectory.terminated,
                    "truncated": trajectory.truncated,
                    "log_probs": trajectory.log_probs,
                    "advantages": advantages,
                    "returns": advantages + values,
                    "values": values,
                },
                transforms=transforms,
            )

            for _ in range(self.k_epoch):
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
                for batch in loader:
                    self.agent.update(batch)

            self.total_episodes += 1

            if after_episode_callback:
                after_episode_callback(self.total_episodes, meta)

    def rollout(self, step: int) -> tuple[RolloutBatch, Any]:
        num_steps = self.num_steps
        num_envs = self.envs.num_envs

        obs, _ = self.envs.reset()  # (num_envs, obs_dim)

        states = torch.zeros(
            (num_steps, num_envs, *self.envs.single_observation_space.shape),  # pyright: ignore[reportOptionalIterable]
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.zeros_like(states)
        rewards = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        terminated_buf = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.int8,
            device=self.device,
        )
        truncated_buf = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.int8,
            device=self.device,
        )
        log_probs = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        raw_actions = torch.zeros(
            (num_steps, num_envs, *self.envs.single_action_space.shape),  # pyright: ignore[reportOptionalIterable]
            dtype=torch.float32,
            device=self.device,
        )

        episode_returns = []
        episode_lengths = []

        state = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for t in range(num_steps):
            action, raw_action, prob = self.agent.get_action(state)

            next_state_np, reward, terminated, truncated, info = self.envs.step(
                action.cpu().numpy()
            )

            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.device)
            bonus = self.agent.reward_bonus(state, next_state)

            states[t] = state
            next_states[t] = next_state
            raw_actions[t] = raw_action
            rewards[t] = torch.tensor(reward + bonus, dtype=torch.float32, device=self.device)
            terminated_buf[t] = torch.tensor(terminated, dtype=torch.int8, device=self.device)
            truncated_buf[t] = torch.tensor(truncated, dtype=torch.int8, device=self.device)
            log_probs[t] = prob

            state = next_state

            # stats
            if "episode" in info:
                mask = info["_episode"]
                for i in range(num_envs):
                    if mask[i]:
                        episode_returns.append(info["episode"]["r"][i])
                        episode_lengths.append(info["episode"]["l"][i])

        return (
            RolloutBatch(
                states,
                next_states,
                raw_actions,
                rewards,
                terminated_buf,
                truncated_buf,
                log_probs,
            ),
            (episode_returns, episode_lengths),
        )

    def advantage(self, rollout: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        T, N = rollout.rewards.shape

        with torch.no_grad():
            values = self.agent.critic(rollout.states.reshape(T * N, -1)).reshape(T, N)
            next_values = self.agent.critic(rollout.next_states.reshape(T * N, -1)).reshape(T, N)

        not_done = 1.0 - rollout.terminated.float()

        deltas = rollout.rewards + self.agent.gamma * next_values * not_done - values

        advantages = torch.zeros_like(rollout.rewards)
        gae = torch.zeros(N, device=rollout.rewards.device)

        for t in reversed(range(T)):
            gae = deltas[t] + self.agent.gamma * self.agent.lamb * not_done[t] * gae
            advantages[t] = gae

        return advantages, values
