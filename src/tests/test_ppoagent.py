from functools import partial

import gymnasium as gym
import pytest

from src.agents.common import norm_advantage
from src.agents.ppo.agent import AgentConfig, PPOAgentContinuous
from src.agents.ppo.envs import make_env
from src.agents.ppo.training import PPOTrainer

ENV_NAME = "LunarLander-v3"
NUM_ENVS = 8

NUM_EPISODES = 1
NUM_STEPS = 1024
K_EPOCH: int = 4
BATCH_SIZE: int = 32


@pytest.fixture
def mock_hyperparms():
    return AgentConfig(
        observable_space=8,
        action_space=2,
        learning_rate=1.5e-3,
        total_updates=NUM_STEPS * NUM_EPISODES * K_EPOCH,
        gamma=0.99,
        lamb=0.9,
    )


@pytest.fixture
def mock_agent(mock_hyperparms):
    return PPOAgentContinuous(mock_hyperparms)


@pytest.fixture
def mock_envs():
    return gym.vector.SyncVectorEnv(
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


def test_mock(mock_agent):
    assert mock_agent is not None


def test_training_loop(mock_agent, mock_envs):
    trainer = PPOTrainer(
        mock_agent,
        mock_envs,
        NUM_STEPS,
        K_EPOCH,
        BATCH_SIZE,
    )
    trainer.train(num_episodes=1, transforms={"advantages": norm_advantage})


# def test_invalid_input():
#     with pytest.raises(TypeError):
