import torch

from src.agents.cart_a2c import A2CCartAgent
from src.agents.common import calc_advantage

EPS = 0.2


class PPOCartAgent(A2CCartAgent):
    def update(self, rewards, probs, critiques: torch.Tensor):
        advantage = calc_advantage(rewards, critiques, self.discount_factor)

        # crutch to make it compatible with a2c api.
        # actual ratio calculated outside of method.
        ratio: torch.Tensor = probs  # pyright: ignore[reportAssignmentType]

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advantage
        loss = -torch.min(surr1, surr2).mean()

        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

    def evaluate_action(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        probs = self.policy.forward(obs)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)
