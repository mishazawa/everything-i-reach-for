import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np

BIN_COUNT = 50


def discrete_value(input: float, bins: list[float], bin_count=BIN_COUNT) -> float:
    return min(np.digitize(input, bins), bin_count - 1)


class CartAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        bins: tuple[list[int], list[int], list[int], list[int]],
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
        self.bins_cartpos = bins[0]
        self.bins_vel = bins[1]
        self.bins_poleangle = bins[2]
        self.bins_angvel = bins[3]

    def get_action(self, obs: tuple[float, float, float, float]) -> int:
        obs = self._prepare_input(obs)
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

        obs = self._prepare_input(obs)
        next_obs = self._prepare_input(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def _prepare_input(
        self,
        obs: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        return (
            discrete_value(obs[0], self.bins_cartpos),
            discrete_value(obs[1], self.bins_vel),
            discrete_value(obs[2], self.bins_poleangle),
            discrete_value(obs[3], self.bins_angvel),
        )

    def decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def cart():
    with open("./data/q_table.pkl", "rb") as f:
        loaded = pickle.load(f)

    Q = defaultdict(lambda: np.zeros(2), loaded)  # 2 = your action count

    env = gym.make("CartPole-v1", render_mode="human")
    SCALE = 10
    SCALE_ANGULAR = 1
    eps = 1e-6

    u = np.linspace(-1 + eps, 1 - eps, num=BIN_COUNT)
    bins_angvel = SCALE_ANGULAR * np.arctanh(u)
    bins_vel = SCALE * np.arctanh(u)

    bins_cartpos = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], num=BIN_COUNT
    )
    bins_poleangle = np.linspace(
        env.observation_space.low[2], env.observation_space.high[2], num=BIN_COUNT
    )

    alpha = 0.75  # 0 = pure arctanh, 1 = pure linear
    linear_part = u
    warped_part = np.arctanh(u) / np.arctanh(1 - eps)  # normalize to similar scale
    bins_angvel = SCALE_ANGULAR * (alpha * linear_part + (1 - alpha) * warped_part)

    agent = CartAgent(
        env=env,
        learning_rate=0,
        initial_epsilon=0,
        epsilon_decay=0,
        final_epsilon=0,
        bins=(bins_cartpos, bins_vel, bins_poleangle, bins_angvel),
    )

    agent.q_values = Q

    try:
        while True:
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            print(f"Episode finished! Total reward: {total_reward}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
