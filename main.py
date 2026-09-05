import numpy as np
import onnxruntime as ort

from src.misc.mujoco_env import make_mujoco_env_watch

ENV_NAME = "Ant-v5"


def main():
    env = make_mujoco_env_watch(ENV_NAME)
    state, _ = env.reset()

    agent = ort.InferenceSession("./data/latest.onnx")
    for i in agent.get_inputs():
        print(i.name, i.shape, i.type)
    for o in agent.get_outputs():
        print(o.name, o.shape, o.type)

    done = False
    total_reward = 0.0
    while not done:
        obs = state[None].astype(np.float32)
        (mu,) = agent.run(None, {"obs": obs})

        state, reward, terminated, truncated, _ = env.step(mu[0])
        done = terminated or truncated
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
