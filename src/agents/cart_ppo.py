import torch

from src.agents.cart_a2c import A2CCartAgent
from src.agents.common import calc_advantage

EPS = 0.2
ENTROPY_COEFF = 0.01


class PPOCartAgent(A2CCartAgent):
    def update(self, rewards, probs, critiques: torch.Tensor):
        advantage = calc_advantage(rewards, critiques, self.discount_factor)

        # crutch to make it compatible with a2c api.
        # actual ratio calculated outside of method.
        ratio: torch.Tensor = probs[0]  # pyright: ignore[reportAssignmentType]
        entropy = probs[1]

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        loss = actor_loss - entropy * ENTROPY_COEFF

        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

    def evaluate_action(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.policy.forward(obs)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy().mean()
