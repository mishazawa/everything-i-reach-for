import numpy as np
import onnxruntime as ort

from utils import make_mujoco_env_watch

ENV_NAME = "Reacher-v5"


def main(checkpoint_path="./data/latest.onnx") -> None:
    env = make_mujoco_env_watch(ENV_NAME)
    state, _ = env.reset()

    agent = ort.InferenceSession(checkpoint_path)
    for i in agent.get_inputs():
        print(i.name, i.shape, i.type)
    for o in agent.get_outputs():
        print(o.name, o.shape, o.type)

    done = False
    total_reward = 0.0
    while not done:
        obs = state[None].astype(np.float32)
        (mu,) = agent.run(None, {"obs": obs})

        state, reward, terminated, truncated, _ = env.step(mu[0])  # pyright: ignore[reportIndexIssue]
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    env.close()
