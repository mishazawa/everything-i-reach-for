from collections import defaultdict
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from src.utils import discrete_value

MAX_LEN = 10_000
BATCH_SIZE = 64
MIN_BUFFER = 1000
SYNC_FREQ = 500
N_PARAMS = 20


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
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay(self):
        self.epsilon = self.decay_fn()


def prepare_input(
    obs: tuple[float, float, float, float],
    bins: tuple[list[float], list[float], list[float], list[float]],
) -> tuple[float, float, float, float]:
    return (
        discrete_value(obs[0], bins[0], len(bins[0])),
        discrete_value(obs[1], bins[1], len(bins[1])),
        discrete_value(obs[2], bins[2], len(bins[2])),
        discrete_value(obs[3], bins[3], len(bins[3])),
    )
