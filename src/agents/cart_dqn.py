import random
from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.networks.dqn import DQN

MAX_LEN = 10_000
BATCH_SIZE = 64
MIN_BUFFER = 1000
SYNC_FREQ = 500
N_PARAMS = 20


class DQNCartAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        decay_fn: Callable[[], float],
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.discount_factor = discount_factor
        self.decay_fn = decay_fn

        self.training_queue = deque(maxlen=MAX_LEN)

        self.q_net = DQN(env.observation_space.shape[0], env.action_space.n, hidden=N_PARAMS)

        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n, hidden=N_PARAMS)

        self.sync_networks(0)
        self.optimizer = Adam(self.q_net.net.parameters(), learning_rate)

    def get_action(self, obs: tuple[float, float, float, float]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            return int(torch.argmax(self.q_net.forward(torch.tensor(obs, dtype=torch.float32))))

    def update(self, batch_size=BATCH_SIZE):
        if len(self.training_queue) < MIN_BUFFER:
            return

        batch = random.sample(self.training_queue, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_pred = self.q_net.forward(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            best_actions = self.q_net.forward(next_states).argmax(1, keepdim=True)
            q_next = self.target_net.forward(next_states).gather(1, best_actions).squeeze(1)
            q_target = rewards + self.discount_factor * q_next * (1 - dones)

        loss = F.mse_loss(q_pred, q_target)

        self.training_error.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def accumulate_train_data(self, state, action, reward, next_state, done):
        self.training_queue.append((state, action, reward, next_state, done))

    def sync_networks(self, step):
        if step % SYNC_FREQ == 0:
            self.target_net.net.load_state_dict(self.q_net.net.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, param in zip(
            self.target_net.net.parameters(), self.q_net.net.parameters(), strict=False
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def decay(self):
        self.epsilon = self.decay_fn()
