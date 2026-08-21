import gymnasium as gym
import onnxruntime as ort

from src.inference.cart_pole import run_inference

CHECKPOINT = "acrobot_ppo_lr0.00025_ne6000_k4_r1_20260821_134336"


def main():
    print("Hello from everything-i-reach-for!")
    env = gym.make("Acrobot-v1", render_mode="human")
    state, _ = env.reset()

    agent = ort.InferenceSession(f"./data/{CHECKPOINT}.onnx")

    done = False
    total_reward = 0
    while not done:
        action = run_inference(agent, state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
