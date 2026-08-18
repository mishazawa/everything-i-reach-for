from torch import nn


class MLP(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)
