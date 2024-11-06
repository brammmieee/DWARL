from gymnasium.wrappers import TimeLimit

class TimeLimitWrapper(TimeLimit):
    def __init__(self, env, cfg):
        super().__init__(env=env, max_episode_steps=int(cfg.max_episode_steps))