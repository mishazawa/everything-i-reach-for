from torch import nn


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
