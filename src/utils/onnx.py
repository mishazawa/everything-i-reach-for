import torch
from tianshou.algorithm.algorithm_base import Algorithm


def save_fn(output, training_envs):
    def sv(policy: Algorithm):
        state = {"model": policy.state_dict(), "obs_rms": training_envs.get_obs_rms()}
        torch.save(state, output)

    return sv


def load_checkpoint(file):
    state = torch.load(file, map_location="cpu", weights_only=False)
    filtered_state = {
        k[len("policy.") :]: v for k, v in state["model"].items() if k.startswith("policy.")
    }
    return filtered_state, state["obs_rms"]


class ONNXActor(torch.nn.Module):
    def __init__(self, actor, obs_rms, max_action=1.0):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", torch.as_tensor(obs_rms.mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(obs_rms.var, dtype=torch.float32))
        self.eps = float(obs_rms.eps)
        self.clip_max = float(obs_rms.clip_max)
        self.max_action = max_action

    def forward(self, obs):
        obs = (obs - self.mean) / torch.sqrt(self.var + self.eps)
        obs = torch.clamp(obs, -self.clip_max, self.clip_max)
        (mu, _), _ = self.actor(obs)
        return torch.clip(mu, -1, 1) * self.max_action


def export_onnx(file, policy, shape, output="./data/latest.onnx"):
    import copy

    policy_copy = copy.deepcopy(policy)

    model_state, obs_rms = load_checkpoint(file)
    policy_copy.load_state_dict(model_state)
    model = ONNXActor(policy_copy.actor, obs_rms).eval()

    torch.onnx.export(
        model,
        torch.randn(1, shape[0]),  # pyright: ignore[reportArgumentType]
        output,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        external_data=False,
    )
