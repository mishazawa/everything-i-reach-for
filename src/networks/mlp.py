import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class MLPBox(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

        self.std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.std)
        return mean, std
