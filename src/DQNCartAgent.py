from collections.abc import Callable

import gymnasium as gym
import numpy as np
from torch import nn


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
        self.qnet = nn.Sequential(
            nn.Linear(4, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 2),
        )

    def get_action(self, obs: tuple[float, float, float, float]) -> int:
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
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay(self):
        self.epsilon = self.decay_fn()
