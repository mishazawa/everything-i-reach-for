import gymnasium as gym
import torch

from src.agents import DQNCartAgent


def cart():
    env = gym.make("CartPole-v1", render_mode="human")

    agent = DQNCartAgent(
        env=env,
        learning_rate=0,
        initial_epsilon=0,
        epsilon_decay=0,
        final_epsilon=0,
        decay_fn=lambda: 0,
    )

    checkpoint = torch.load("./data/checkpoint_6000.pt")

    agent.q_net.load_state_dict(checkpoint["q_net"])
    agent.target_net.load_state_dict(checkpoint["q_net"])

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
