import random
from collections import defaultdict, deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from src.utils import discrete_value

MAX_LEN = 10_000
BATCH_SIZE = 64
MIN_BUFFER = 1000
SYNC_FREQ = 500
N_PARAMS = 20


class DQNCartAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        decay_fn: Callable[[], float],
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.discount_factor = discount_factor
        self.decay_fn = decay_fn

        self.training_queue = deque(maxlen=MAX_LEN)

        self.q_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], N_PARAMS),
            nn.LeakyReLU(),
            nn.Linear(N_PARAMS, env.action_space.n),
        )

        self.target_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], N_PARAMS),
            nn.LeakyReLU(),
            nn.Linear(N_PARAMS, env.action_space.n),
        )

        self.sync_networks(0)
        self.optimizer = Adam(self.q_net.parameters(), learning_rate)

    def get_action(self, obs: tuple[float, float, float, float]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            return int(torch.argmax(self.q_net(torch.tensor(obs, dtype=torch.float32))))

    def update(self, batch_size=BATCH_SIZE):
        if len(self.training_queue) < MIN_BUFFER:
            return

        batch = random.sample(self.training_queue, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_pred = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            q_target = rewards + self.discount_factor * q_next * (1 - dones)

        loss = F.mse_loss(q_pred, q_target)

        self.training_error.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def accumulate_train_data(self, state, action, reward, next_state, done):
        self.training_queue.append((state, action, reward, next_state, done))

    def sync_networks(self, step):
        if step % SYNC_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def decay(self):
        self.epsilon = self.decay_fn()


class QartAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        bins: tuple[list[float], list[float], list[float], list[float]],
        decay_fn: Callable[[], float],
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.discount_factor = discount_factor
        self.decay_fn = decay_fn
        self.bins = bins

    def get_action(self, obs: tuple[float, float, float, float]) -> int:
        obs = prepare_input(obs, self.bins)
        return (
            self.env.action_space.sample()
            if np.random.random() < self.epsilon
            else int(np.argmax(self.q_values[obs]))
        )

    def update(
        self,
        obs: tuple[float, float, float, float],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[float, float, float, float],
    ):

        obs = prepare_input(obs, self.bins)
        next_obs = prepare_input(next_obs, self.bins)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay(self):
        self.epsilon = self.decay_fn()


def prepare_input(
    obs: tuple[float, float, float, float],
    bins: tuple[list[float], list[float], list[float], list[float]],
) -> tuple[int, int, int, int]:
    return (
        discrete_value(obs[0], bins[0], len(bins[0])),
        discrete_value(obs[1], bins[1], len(bins[1])),
        discrete_value(obs[2], bins[2], len(bins[2])),
        discrete_value(obs[3], bins[3], len(bins[3])),
    )
