import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.agents.cart_ppo import PPOCartAgent

EPS = 1e-6


@dataclass
class HP:
    num_episodes: int
    repeat: int
    k_epoch: int


def create_linear_bin(high, low, bc):
    return np.linspace(low, high, num=bc)


def create_nonlinear_bin(bc):
    u = np.linspace(-1 + EPS, 1 - EPS, num=bc)
    return np.arctanh(u)


def rectify_nonlinear_bin(bc, alpha=0.75):
    u = np.linspace(-1 + EPS, 1 - EPS, num=bc)
    linear_part = u
    warped_part = np.arctanh(u) / np.arctanh(1 - EPS)  # normalize to similar scale
    return alpha * linear_part + (1 - alpha) * warped_part


def scale_bin(b, s):
    return b * s


def discrete_value(input: float, bins: list[float], bin_count) -> float:
    return np.min(np.digitize(input, bins), bin_count - 1)


def create_bins_for_use(env, bc, scale):
    BIN_COUNT = bc
    SCALE_VEL, SCALE_ANG = scale

    bins_cartpos = create_linear_bin(
        env.observation_space.high[0], env.observation_space.low[0], BIN_COUNT
    )
    bins_poleangle = create_linear_bin(
        env.observation_space.high[2], env.observation_space.low[2], BIN_COUNT
    )
    bins_vel = scale_bin(create_nonlinear_bin(BIN_COUNT), SCALE_VEL)
    bins_angvel = scale_bin(rectify_nonlinear_bin(BIN_COUNT), SCALE_ANG)

    return (bins_cartpos, bins_vel, bins_poleangle, bins_angvel)


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode) / window


def exponential_decay(start_epsilon, final_epsilon, epsilon_decay):
    epsilon = start_epsilon

    def calc():
        nonlocal epsilon
        epsilon = max(final_epsilon, epsilon * epsilon_decay)
        return epsilon

    return calc


def linear_decay(start_epsilon, final_epsilon, epsilon_decay):
    epsilon = start_epsilon

    def calc():
        nonlocal epsilon
        epsilon = max(final_epsilon, epsilon - epsilon_decay)
        return epsilon

    return calc


def train_ppo(agent: PPOCartAgent, env: gym.Env, writer: SummaryWriter, hp: HP):
    for e in range(hp.num_episodes):
        state, _ = env.reset()

        done = False
        states = []

        # FIRST PASS:
        actions = []
        probs = []
        rewards = []
        running_batch = []
        t = 0
        while not done:
            states.append(state)

            action, prob = agent.get_action(state)

            # bound vars
            next_state = state
            reward = 0

            total_reward = 0
            for _ in range(hp.repeat):
                next_state, reward, terminated, truncated, _ = env.step(action)
                t += 1

                total_reward += reward

                done = terminated or truncated
                if done:
                    break

            probs.append(prob)
            actions.append(action)
            rewards.append(total_reward)

            if agent.update_critic(running_batch):
                running_batch = []

            running_batch.append((state, action, reward, next_state, done))
            state = next_state

        # TRAINING PASS:
        old_log_probs = torch.stack(probs).detach()
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int8)

        for _ in range(hp.k_epoch):
            with torch.no_grad():
                critiques = agent.critic(states_t).squeeze(-1)
            new_log_probs = agent.evaluate_action(states_t, actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs)
            agent.update(rewards, ratio, critiques)  # pyright: ignore[reportArgumentType]

        writer.add_scalar("duration", t, e + 1)


def run_tests(agent: PPOCartAgent, env: gym.Env, writer: SummaryWriter, num_tests=1000):
    for e in range(num_tests):
        state, info = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            state = next_state

        writer.add_scalar("eval/reward", info["episode"]["r"], e + 1)

    data = list(env.return_queue)
    win_rate = np.mean(np.array(data) > 0)
    average_reward = np.mean(data)
    return average_reward, np.min(data), np.max(data), win_rate


def save_checkpoint(
    state_dict: dict,
    epoch: int,
    reward: float,
    checkpoint_dir: str = "./data/checkpoints",
    ttl_epochs: int = 100,
    best_reward_tracker: list | None = None,
) -> bool:
    """
    Saves a model checkpoint based on TTL (epoch frequency) or best reward performance.

    Args:
        state_dict (dict): Dictionary containing model/optimizer states and metadata.
        epoch (int): Current training epoch.
        reward (float): Metric/reward achieved in the current epoch.
        checkpoint_dir (str): Directory where checkpoints will be saved.
        ttl_epochs (int): Save frequency interval (e.g., save every N epochs).
        best_reward_tracker (list): Single-element list used as a mutable container
                                    to track the highest reward across function calls.

    Returns:
        bool: True if a checkpoint was saved, False otherwise.
    """
    if best_reward_tracker is None:
        best_reward_tracker = [-float("inf")]
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved = False

    # Check conditions
    is_ttl_match = (epoch % ttl_epochs == 0) and (epoch > 0)
    is_best_reward = reward > best_reward_tracker[0]

    # 1. Save Best Reward Checkpoint
    if is_best_reward:
        best_reward_tracker[0] = reward
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

        save_payload = {
            **state_dict,
            "epoch": epoch,
            "reward": reward,
        }
        torch.save(save_payload, best_path)
        print(f"[Checkpoint] New best reward achieved ({reward:.4f}). Saved to {best_path}")
        saved = True

    # 2. Save Interval (TTL) Checkpoint
    if is_ttl_match:
        ttl_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        save_payload = {
            **state_dict,
            "epoch": epoch,
            "reward": reward,
        }
        torch.save(save_payload, ttl_path)
        print(f"[Checkpoint] TTL interval reached (Epoch {epoch}). Saved to {ttl_path}")
        saved = True

    return saved


def load_checkpoint(file):
    state = torch.load(file, map_location="cpu", weights_only=False)
    filtered_state = {
        k[len("policy.") :]: v for k, v in state["model"].items() if k.startswith("policy.")
    }
    return filtered_state, state["obs_rms"]


class ONNXActor(torch.nn.Module):
    def __init__(self, actor, obs_rms, max_action=1.0):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", torch.as_tensor(obs_rms.mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(obs_rms.var, dtype=torch.float32))
        self.eps = float(obs_rms.eps)
        self.clip_max = float(obs_rms.clip_max)
        self.max_action = max_action

    def forward(self, obs):
        obs = (obs - self.mean) / torch.sqrt(self.var + self.eps)
        obs = torch.clamp(obs, -self.clip_max, self.clip_max)
        (mu, _), _ = self.actor(obs)
        return torch.clip(mu, -1, 1) * self.max_action


def export_onnx(file, policy, shape, output="./data/latest.onnx"):
    import copy

    policy_copy = copy.deepcopy(policy)

    model_state, obs_rms = load_checkpoint(file)
    policy_copy.load_state_dict(model_state)
    model = ONNXActor(policy_copy.actor, obs_rms).eval()

    torch.onnx.export(
        model,
        torch.randn(1, shape[0]),  # pyright: ignore[reportArgumentType]
        output,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        external_data=False,
    )
