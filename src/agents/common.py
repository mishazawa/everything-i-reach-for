import torch


def calc_advantage(rewards, critiques, initial_g=0, gamma=0.95):
    ret = []
    G = initial_g
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)

    ret = torch.tensor(ret)

    return ret - critiques


def calc_gae(rewards, critiques, initial_v=0, gamma=0.95, lam=0.95):
    ret = []
    A = 0
    next_v = initial_v
    for r, v in zip(reversed(rewards), reversed(critiques), strict=False):
        delta = r + gamma * next_v - v
        A = delta + gamma * lam * A
        ret.insert(0, A)
        next_v = v

    return torch.tensor(ret)


def norm_advantage(advantage: torch.Tensor):
    return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
