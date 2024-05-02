from stable_baselines3.common.env_checker import check_env
from environments.custom_env import CustomEnv

env = CustomEnv(render_mode=None, wb_open=True, wb_mode='training')
check_env(env)