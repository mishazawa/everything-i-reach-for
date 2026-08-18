import gymnasium as gym
import torch
from safetensors.torch import load_file
from torch import nn

HIDDEN = 128


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def load_agent(weights_path, n_observations, n_actions, device="cpu"):
    agent = DQN(n_observations, n_actions).to(device)
    state_dict = load_file(weights_path, device=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def get_action(agent, state, device="cpu"):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return agent(state_t).max(1).indices.item()


if __name__ == "__main__":
    device = "cpu"
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()

    n_observations = len(state)
    n_actions = env.action_space.n

    agent = DQN(n_observations, n_actions, hidden=20).to(device)

    state_dict = load_file("./data/checkpoint_6000.safetensors", device=device)

    state_dict = {
        f"net.{k}": v for k, v in state_dict.items()
    }  # crutch because of DQNAgent naming

    agent.load_state_dict(state_dict)
    agent.eval()

    done = False
    total_reward = 0
    while not done:
        action = get_action(agent, state, device)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")

    env.close()
