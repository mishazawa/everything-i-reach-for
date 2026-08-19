import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from src.networks.dqn import DQN
from src.networks.mlp import MLP

MIN_BUFFER = 64


class A2CCartAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.discount_factor = discount_factor

        self.policy = MLP(env.observation_space.shape[0], env.action_space.n)
        self.critic = DQN(env.observation_space.shape[0], 1, 20)

        self.optimizer_policy = Adam(self.policy.net.parameters(), learning_rate)
        self.optimizer_critic = Adam(self.critic.net.parameters(), learning_rate)

    def get_action(self, obs: tuple[float, float, float, float]) -> tuple[int, float]:
        state = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def update_critic(self, batch) -> bool:
        if len(batch) < MIN_BUFFER:
            return False

        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_pred = self.critic.forward(states).squeeze(1)
        with torch.no_grad():
            q_next = self.critic.forward(next_states).squeeze(1)
            q_target = rewards + self.discount_factor * q_next * (1 - dones)

        loss = F.mse_loss(q_pred, q_target)

        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        return True

    def update(self, rewards: list[int], probs: list[float], critiques):
        ret = []

        G = 0

        for r in reversed(rewards):
            G = r + self.discount_factor * G
            ret.insert(0, G)

        ret = torch.tensor(ret)

        advantage = ret - critiques

        loss = torch.stack([-lp * G for lp, G in zip(probs, advantage, strict=False)]).sum()

        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

    @torch.no_grad()
    def act(self, obs: tuple[float, float, float, float], deterministic: bool = True) -> int:
        state = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy.forward(state)
        if deterministic:
            action = torch.argmax(probs)
        else:
            action = torch.distributions.Categorical(probs).sample()
        return int(action.item())

    @torch.no_grad()
    def get_critique(self, x) -> float:
        return self.critic.forward(x)
