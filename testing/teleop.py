# %%
%matplotlib qt

# %%
from inputs import get_gamepad
import numpy as np
from environments.base_env import BaseEnv

# %%
env = BaseEnv(
    render_mode='full', 
    wb_open=True, 
    wb_mode='testing', 
    proto_config='sparse_lidar_proto_config.json', 
    teleop=True
)

env.reset()

n_steps = 100000000
for i in range(n_steps):
    # action = env.action_space.sample()
    action = np.array([-1.0, 1.0])
    # linear_vel, angular_vel = read_gamepad_input()
    # action = np.array([angular_vel, linear_vel])

    obs, reward, done, truncated, info = env.step(action)
    # time.sleep(0.4)
    if done:
        obs = env.reset()
        
# %%