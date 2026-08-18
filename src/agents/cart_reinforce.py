import gymnasium as gym
import torch
from torch.optim.adam import Adam

from src.networks.mlp import MLP


class ReinforceCartAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.training_error = []
        self.discount_factor = discount_factor

        self.policy = MLP(env.observation_space.shape[0], env.action_space.n)

        self.optimizer = Adam(self.policy.net.parameters(), learning_rate)

    def get_action(self, obs: tuple[float, float, float, float]) -> tuple[int, float]:
        state = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def update(self, rewards: list[int], probs: list[float]):
        ret = []

        G = 0

        for i in reversed(rewards):
            G = i + self.discount_factor * G
            ret.insert(0, G)

        ret = torch.tensor(ret)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        loss = torch.stack([-lp * G for lp, G in zip(probs, ret, strict=False)]).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def act(self, obs: tuple[float, float, float, float], deterministic: bool = True) -> int:
        state = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy.forward(state)
        if deterministic:
            action = torch.argmax(probs)
        else:
            action = torch.distributions.Categorical(probs).sample()
        return int(action.item())
