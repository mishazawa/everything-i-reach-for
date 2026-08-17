import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np

from src.CartAgent import CartAgent as QCartAgent
from src.utils import create_bins_for_use


def cart():
    with open("./data/q_table_0.1_200000_33_2_1_0.99_exp.pkl", "rb") as f:
        loaded = pickle.load(f)

    Q = defaultdict(lambda: np.zeros(2), loaded)  # 2 = your action count

    env = gym.make("CartPole-v1", render_mode="human")

    agent = QCartAgent(
        env=env,
        learning_rate=0,
        initial_epsilon=0,
        epsilon_decay=0,
        final_epsilon=0,
        bins=create_bins_for_use(env, 33, (2, 1)),
        decay_fn=lambda: 0,
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
