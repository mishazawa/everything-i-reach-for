from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.LeakyReLU(),
            # nn.Linear(hidden, hidden),
            # nn.LeakyReLU(),
        )

        self.value_stream = nn.Linear(hidden, 1)

        self.advantage_stream = nn.Linear(hidden, n_actions)

    def forward(self, x):
        f = self.net(x)
        value = self.value_stream(f)
        advantage = self.advantage_stream(f)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
