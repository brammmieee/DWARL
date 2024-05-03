# %%
%matplotlib qt

# %%
import numpy as np
from environments.base_env import BaseEnv
from environments.wrappers.obstacle_velocity_observation_wrapper import ObstacleVelocityObservationWrapper as OVOWrapper

# %%
# env = BaseEnv(render_mode='full', wb_open=True, wb_mode='testing', reward_monitoring=False)

# %%
env = OVOWrapper(BaseEnv(render_mode='full', wb_open=True, wb_mode='testing', reward_monitoring=False))

# %%
obs = env.reset() #options={"map_nr":40, "nominal_dist":1})

# %%
action = np.array([1.0, 1.0])
obs, reward, done, _, _ = env.step(action)

# %%
import time

# env.set_render_mode('velocity')
# obs = env.reset()
n_steps = 100000000
for i in range(n_steps):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # time.sleep(0.4)
    if done:
        obs = env.reset()