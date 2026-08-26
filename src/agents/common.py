import torch


def calc_advantage(rewards, critiques, initial_g=0, gamma=0.95):
    ret = []
    G = initial_g
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)

    ret = torch.tensor(ret)

    return ret - critiques


def norm_advantage(advantage: torch.Tensor):
    return (advantage - advantage.mean()) / (advantage.std() + 1e-8)
