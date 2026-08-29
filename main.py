import gymnasium as gym
import numpy as np
import onnxruntime as ort

CHECKPOINT = "latest"

# ENV_NAME = "MountainCarContinuous-v0"
ENV_NAME = "LunarLander-v3"


def main():
    print("Hello from everything-i-reach-for!")
    env = gym.make(
        ENV_NAME,
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="human",
    )
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
    from datetime import datetime
    from functools import partial

    import gymnasium as gym
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from src.agents.common import norm_advantage
    from src.agents.ppo.agent import AgentConfig, PPOAgentContinuous
    from src.agents.ppo.envs import make_env
    from src.agents.ppo.training import PPOTrainer

    run_name = f"ppo_ne_{datetime.now()}"
    writer = SummaryWriter(f"./logs/{run_name}")

    def after_ep(t, meta):
        episode_returns, ep_len = meta
        if len(episode_returns):
            sum_reward = float(sum(episode_returns) / len(episode_returns))
            sum_ep_len = float(sum(ep_len) / len(episode_returns))
        else:
            sum_reward = 0
            sum_ep_len = 0

        writer.add_scalar("train/reward", sum_reward, t)
        writer.add_scalar("train/len", sum_ep_len, t)

    ENV_NAME = ENV_NAME = "LunarLander-v3"
    NUM_ENVS = 8

    NUM_EPISODES = 1000
    NUM_STEPS = 1024
    K_EPOCH: int = 4
    BATCH_SIZE: int = 32

    device = torch.device("cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            partial(
                make_env,
                ENV_NAME,
                continuous=True,
                gravity=-10.0,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5,
                render_mode=None,
            )
            for _ in range(NUM_ENVS)
        ]
    )

    agent = PPOAgentContinuous(
        AgentConfig(
            observable_space=envs.single_observation_space.shape[0],
            action_space=envs.single_action_space.shape[0],
            learning_rate=1.5e-3,
            total_updates=NUM_STEPS * NUM_EPISODES * K_EPOCH,
            gamma=0.99,
            lamb=0.9,
            device=device,
        )
    )

    trainer = PPOTrainer(
        agent,
        envs,
        device,
        NUM_STEPS,
        K_EPOCH,
        BATCH_SIZE,
    )

    trainer.train(
        num_episodes=NUM_EPISODES,
        transforms={"advantages": norm_advantage},
        after_episode_callback=after_ep,
    )
