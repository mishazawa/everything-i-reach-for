import gymnasium as gym
import numpy as np
import onnxruntime as ort

CHECKPOINT = "latest"


def main():
    print("Hello from everything-i-reach-for!")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    state, _ = env.reset()

    agent = ort.InferenceSession(f"./data/{CHECKPOINT}.onnx")

    done = False
    total_reward = 0
    while not done:
        obs = np.array([state], dtype=np.float32)
        mean, _ = agent.run(None, {"obs": obs})
        state, reward, terminated, truncated, _ = env.step(mean[0])  # pyright: ignore[reportIndexIssue]
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
