import numpy as np
import gymnasium as gym

class NoDynamicWindowAction(gym.ActionWrapper):
    '''
    This enviroment wrapper does the following:
    - Projects the action output of the policy into the static velocity window (defined by x_min and x_max).
    '''
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self.env_cfg = self.unwrapped.cfg

    def scale_x(self, x, y_min, y_max, x_min=-1.0, x_max=1.0):
        # Scale the value x that was in the range [x_min, x_max] to the range [y_min, y_max]
        a = (y_max - y_min) / (x_max - x_min)
        b = y_min - a * x_min
        y = a * x + b

        return y

    def map_normalized_action_to_cmd_vel(self, action):
        # No dynamic velocity window
        omega_min = self.env_cfg.vehicle.kinematics.omega_min
        omega_max = self.env_cfg.vehicle.kinematics.omega_max
        v_max = self.env_cfg.vehicle.kinematics.v_max
        v_min = self.env_cfg.vehicle.kinematics.v_min

        omega = self.scale_x(action[0], omega_min, omega_max, -1.0, 1.0)
        v = self.scale_x(action[1], v_min, v_max, -1.0, 1.0)
        cmd_vel = np.array([omega, v])

        return cmd_vel

    def action(self, act):
        cmd_vel = self.map_normalized_action_to_cmd_vel(act)
        return cmd_vel # = action