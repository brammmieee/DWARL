from gymnasium.wrappers import TimeLimit

class TimeLimitWrapper(TimeLimit):
    def __init__(self, env, cfg):
        super().__init__(env, max_episode_steps=cfg.max_episode_steps)