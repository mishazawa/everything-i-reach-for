import gymnasium as gym
import torch
from safetensors.torch import load_file

from src.networks.dqn import DQN
from src.networks.mlp import MLP


def load_agent(weights_path, n_observations, n_actions, device="cpu"):
    agent = DQN(n_observations, n_actions).to(device)
    state_dict = load_file(weights_path, device=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def load_agent_a2c(weights_path, n_observations, n_actions, device="cpu"):
    agent = MLP(n_observations, n_actions).to(device)
    state_dict = load_file(weights_path, device=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


@torch.no_grad
def get_action(agent, state, device="cpu"):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return agent(state_t).max(1).indices.item()


@torch.no_grad
def get_action_a2c(agent, state, device="cpu"):

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    probs = agent(state)
    action = torch.argmax(probs)

    return int(action.item())


if __name__ == "__main__":
    device = "cpu"
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()

    n_observations = len(state)
    n_actions = env.action_space.n

    agent = DQN(n_observations, n_actions, hidden=20).to(device)

    state_dict = load_file("./data/checkpoint_6000.safetensors", device=device)

    state_dict = {f"net.{k}": v for k, v in state_dict.items()}  # crutch because of DQNAgent naming

    agent.load_state_dict(state_dict)
    agent.eval()

    done = False
    total_reward = 0.0
    while not done:
        action = get_action(agent, state, device)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")

    env.close()
