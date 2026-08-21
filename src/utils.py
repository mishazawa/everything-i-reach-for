from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from safetensors.torch import save_file
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


def save_checkpoint(network, name):
    save_file(network.state_dict(), f"./data/{name}.safetensors")


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

    return np.average(list(env.return_queue))
