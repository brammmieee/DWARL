# %%
%matplotlib qt

# %%
import numpy as np
from environments.base_env import BaseEnv
from environments.wrappers.velocity_obstacle_wrapper import VelocityObstacleWrapper as VOWrapper
from environments.wrappers.dynamic_window_action_wrapper import DynamicWindowActionWrapper as DWAActWrapper

# %%
env = BaseEnv(render_mode='full', wb_open=True, wb_mode='testing')

# %%
env = DWAActWrapper(VOWrapper(BaseEnv(render_mode='full', wb_open=True, wb_mode='testing')))

# %%
obs = env.reset() #options={"map_nr":40, "nominal_dist":1})

# %%
action = np.array([-1.0, 1.0])
obs, reward, done, _, _ = env.step(action)

# %%
action = env.action_space.sample()
obs, reward, done, _, _ = env.step(action)

# %%
import time

# env.set_render_mode('velocity')
# obs = env.reset()
n_steps = 100000000
for i in range(n_steps):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    # time.sleep(0.4)
    if done:
        obs = env.reset()

# %%
import utils.admin_tools as at
params = at.load_parameters(["base_parameters.yaml", "sparse_lidar_proto_config.json"]) #TODO: list can be directly parsed to init of base env

# %%
