from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from src.agents.ppo.loss import critic_loss, policy_loss
from src.networks.dqn import DQN
from src.networks.mlp import MLPBox


@dataclass
class AgentConfig:
    observable_space: int
    action_space: int
    learning_rate: float
    total_updates: int
    device: torch.device
    gamma: float = 0.99
    lamb: float = 0.9
    entropy_coef: float = 0.01
    learning_rate_decay: Callable[[int], float] = lambda x: x


class PPOAgent(ABC):
    def __init__(self, data: AgentConfig):
        self.policy = self.init_policy(data)
        self.critic = self.init_critic(data)
        self.optimizer_policy = torch.optim.Adam(self.policy.net.parameters(), data.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.net.parameters(), data.learning_rate)

        self.scheduler_policy = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_policy, data.learning_rate_decay
        )

        self.scheduler_critic = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_critic, data.learning_rate_decay
        )

        self.gamma = data.gamma
        self.lamb = data.lamb
        self.entropy_coef = data.entropy_coef
        self.device = data.device

    def reward_bonus(self, prev_state, next_state) -> float:
        return 0

    def update(self, batch):
        PPOAgent._upd_step(
            critic_loss(self.critic, batch),
            self.optimizer_critic,
            self.scheduler_critic,
        )
        PPOAgent._upd_step(
            policy_loss(self, batch, self.entropy_coef),
            self.optimizer_policy,
            self.scheduler_policy,
        )

    @staticmethod
    def _upd_step(loss, optimizer, scheduler):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    @abstractmethod
    def init_policy(self, data: AgentConfig) -> nn.Module: ...

    @abstractmethod
    def init_critic(self, data: AgentConfig) -> nn.Module: ...

    @abstractmethod
    def get_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def eval_action(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class PPOAgentContinuous(PPOAgent):
    def init_policy(self, data: AgentConfig):
        return MLPBox(data.observable_space, data.action_space).to(data.device)

    def init_critic(self, data: AgentConfig) -> nn.Module:
        return DQN(data.observable_space, 1, 20).to(data.device)

    def get_action(self, state: torch.Tensor):
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return torch.clamp(action, -1.0, 1.0), action, dist.log_prob(action).sum(-1).detach()

    def eval_action(self, state: torch.Tensor, action: torch.Tensor):
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        new_probs = dist.log_prob(action).sum(-1)
        return new_probs, dist.entropy().mean()
