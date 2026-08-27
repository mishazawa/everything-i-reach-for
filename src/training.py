import itertools
import random
from dataclasses import dataclass

import gymnasium as gym
import torch

from src.agents.common import calc_advantage, norm_advantage

BATCH_SIZE = 64
EPS = 0.2
ENTROPY_COEFF = 0.01


@dataclass
class RolloutBatch:
    states: torch.Tensor  # (T, N, obs_dim)
    next_states: torch.Tensor  # (T, N, obs_dim) — patched with real final obs, see below
    raw_actions: torch.Tensor  # (T, N, action_dim) or (T, N) for discrete
    rewards: torch.Tensor  # (T, N)
    terminated: torch.Tensor  # (T, N)
    truncated: torch.Tensor  # (T, N)
    log_probs: torch.Tensor  # (T, N)


def make_batches(data, n=BATCH_SIZE):
    new_data = list(data)
    random.shuffle(new_data)
    return itertools.batched(new_data, n, strict=False)


@torch.no_grad
def get_critiques(critic: torch.nn.Module, observations: torch.Tensor):
    return critic(observations).squeeze(-1)


def compute_advantage(critic: torch.nn.Module, data, gamma):
    batch = list(*zip(*data, strict=False))
    states = torch.stack(batch[0])
    rewards = torch.stack(batch[3])

    critiques = get_critiques(critic, states)

    terminated = batch[4][-1]
    truncated = batch[5][-1]

    if truncated and not terminated:
        with torch.no_grad():
            next_state = batch[1][-1]
            bootstrap = critic(next_state)
    else:
        bootstrap = 0

    return calc_advantage(rewards, critiques, bootstrap, gamma)


def compute_advantages(critic: torch.nn.Module, rollout: RolloutBatch, gamma: float, lam: float):
    T, N = rollout.rewards.shape

    with torch.no_grad():
        values = critic(rollout.states.reshape(T * N, -1)).reshape(T, N)
        next_values = critic(rollout.next_states.reshape(T * N, -1)).reshape(T, N)

    not_done = 1.0 - rollout.terminated.float()

    deltas = rollout.rewards + gamma * next_values * not_done - values

    advantages = torch.zeros_like(rollout.rewards)
    gae = torch.zeros(N)

    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * not_done[t] * gae
        advantages[t] = gae

    return advantages, values


def evaluate_action_continuous(
    policy: torch.nn.Module, observations: torch.Tensor, actions: torch.Tensor
):
    mean, std = policy(observations)
    dist = torch.distributions.Normal(mean, std)
    action = dist.log_prob(actions).sum(-1)
    return action, dist.entropy().mean()


def get_action_continuous(policy: torch.nn.Module, observations: torch.Tensor):
    mean, std = policy(observations)
    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    return action, dist.log_prob(action).sum(-1).detach()


def update_critic(model, batch, clip_eps=EPS):
    states = batch["states"]
    returns = batch[
        "returns"
    ]  # advantages + values, from compute_advantages — fixed for this rollout
    old_values = batch["values"]  # also from compute_advantages, pre-update

    v_pred = model(states).squeeze(-1)
    v_clipped = old_values + torch.clamp(v_pred - old_values, -clip_eps, clip_eps)

    loss_unclipped = (v_pred - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


def update_policy(batch, eval_fn, entropy_coeff=ENTROPY_COEFF):
    states = batch["states"]
    actions = batch["raw_actions"]
    old_log_probs = batch["log_probs"]
    advantage = batch["advantages"]

    new_log_probs, entropy = eval_fn(states, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advantage
    actor_loss = -torch.min(surr1, surr2).mean()

    return actor_loss - entropy * entropy_coeff


def rollout_trajectory(
    envs: gym.vector.VectorEnv,
    action_fn,
    reward_bonus_fn,
    num_steps: int,
):
    num_envs = envs.num_envs
    state = torch.tensor(envs.reset()[0], dtype=torch.float32)  # (num_envs, obs_dim)

    states = torch.zeros((num_steps, num_envs, *state.shape[1:]), dtype=torch.float32)
    next_states = torch.zeros_like(states)
    rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    terminated_buf = torch.zeros((num_steps, num_envs), dtype=torch.int8)
    truncated_buf = torch.zeros((num_steps, num_envs), dtype=torch.int8)
    log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    raw_actions = None

    episode_returns = []
    episode_lengths = []

    for t in range(num_steps):
        action, raw_action, prob = action_fn(state)
        if raw_actions is None:
            raw_actions = torch.zeros(
                (num_steps, num_envs, *raw_action.shape[1:]),
                dtype=raw_action.dtype,
            )

        next_state_np, reward, terminated, truncated, info = envs.step(action.numpy())

        if "final_observation" in info:
            mask = info["_final_observation"]
            for i in range(num_envs):
                if mask[i]:
                    print("111111")
                    next_state_np[i] = info["final_observation"][i]

        next_state = torch.tensor(next_state_np, dtype=torch.float32)
        bonus = reward_bonus_fn(state, next_state)

        states[t] = state
        next_states[t] = next_state
        raw_actions[t] = raw_action
        rewards[t] = torch.tensor(reward + bonus, dtype=torch.float32) + bonus
        terminated_buf[t] = torch.tensor(terminated, dtype=torch.int8)
        truncated_buf[t] = torch.tensor(truncated, dtype=torch.int8)
        log_probs[t] = prob

        state = next_state
        if "episode" in info:
            mask = info["_episode"]  # bool array, True where an episode just ended
            for i in range(num_envs):
                if mask[i]:
                    episode_returns.append(info["episode"]["r"][i])
                    episode_lengths.append(info["episode"]["l"][i])

    return (
        RolloutBatch(
            states,
            next_states,
            raw_actions,  # pyright: ignore[reportArgumentType]
            rewards,
            terminated_buf,
            truncated_buf,
            log_probs,
        ),
        episode_returns,
        episode_lengths,
    )


def assemble_dataset(
    rollout,
    advantages,
    values,
):
    T, N = rollout.rewards.shape
    dataset_size = T * N

    flat = {
        k: v.reshape(dataset_size, *v.shape[2:])
        for k, v in {
            "states": rollout.states,
            "next_states": rollout.next_states,
            "raw_actions": rollout.raw_actions,
            "rewards": rollout.rewards,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "log_probs": rollout.log_probs,
            "advantages": norm_advantage(advantages),
            "returns": advantages + values,
            "values": values,
        }.items()
    }

    return flat, dataset_size


def train(
    dataset,
    dataset_size,
    update_critic_fn,
    update_policy_fn,
):
    indices = torch.randperm(dataset_size)
    for start in range(0, dataset_size, BATCH_SIZE):
        batch_idx = indices[start : start + BATCH_SIZE]
        batch = {k: v[batch_idx] for k, v in dataset.items()}

        update_critic_fn(batch)
        update_policy_fn(batch)
