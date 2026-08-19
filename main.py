import gymnasium as gym

from src.inference.cart_pole import get_action_a2c, load_agent_a2c


def main():
    print("Hello from everything-i-reach-for!")
    device = "cpu"
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()

    n_observations = len(state)
    n_actions = env.action_space.n

    agent = load_agent_a2c(
        "./data/a2c_e6000.safetensors",
        n_observations=n_observations,
        n_actions=n_actions,
        device=device,
    )

    done = False
    total_reward = 0
    while not done:
        action = get_action_a2c(agent, state, device)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
