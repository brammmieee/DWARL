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








# %% ONLY USE ICW A GAMEPAD TODO: Move to the base tools
def map_stick_to_velocity(stick_value, max_velocity):
    # Since ABS_HAT0X and ABS_HAT0Y values are -1, 0, or 1, we don't need to normalize
    return stick_value * max_velocity

def read_gamepad_input():
    linear_velocity_max = 1.0  # Adjust as needed
    angular_velocity_max = 1.0  # Adjust as needed
    linear_velocity = 0
    angular_velocity = 0

    events = get_gamepad()
    for event in events:
        if event.ev_type == 'Absolute':
            if event.code == 'ABS_HAT0Y':  # Use HAT0Y for linear velocity
                linear_velocity = map_stick_to_velocity(-event.state, linear_velocity_max)  # Invert if necessary
            elif event.code == 'ABS_HAT0X':  # Use HAT0X for angular velocity
                angular_velocity = map_stick_to_velocity(event.state, angular_velocity_max)

    return linear_velocity, angular_velocity
