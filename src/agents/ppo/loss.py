import torch
import torch.nn.functional as F

PPO_EPS = 0.2


def policy_loss(agent, batch, entropy_coeff):
    states = batch["states"]
    actions = batch["raw_actions"]
    old_log_probs = batch["log_probs"]
    advantage = batch["advantages"]

    new_log_probs, entropy = agent.eval_action(states, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * advantage
    actor_loss = -torch.min(surr1, surr2).mean()

    return actor_loss - entropy * entropy_coeff


def critic_loss(model: torch.nn.Module, batch) -> torch.Tensor:
    states = batch["states"]
    returns = batch["returns"]

    v_pred = model(states).squeeze(-1)

    return 0.5 * F.mse_loss(v_pred, returns)
