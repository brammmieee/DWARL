from gymnasium.wrappers import TimeLimit

def apply_time_limit(env):
    # Add time limit wrapper to limit the episode length without breaking Markov assumption
    params = at.load_parameters(['base_parameters.yaml'])
    max_episode_steps = params['max_ep_time'] / params['sample_time']
    return TimeLimit(env, max_episode_steps=max_episode_steps)