import gymnasium as gym


def make_env(env_name, **kwargs):
    env = gym.make(id=env_name, **kwargs)
    return gym.wrappers.RecordEpisodeStatistics(env)
