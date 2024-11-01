import numpy as np
import gymnasium as gym

class CommandVelocityAction(gym.ActionWrapper):
    '''
    This enviroment wrapper does the following:
    - Projects the action output of the policy into the dynamic velocity window.
    - Clips the action to the maximal velocity bounds.
    '''
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self.env_cfg = self.unwrapped.cfg

        # Define action space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    def map_normalized_action_to_cmd_vel(self, action):
        omega_min = self.env_cfg.vehicle.kinematics.omega_min
        omega_max = self.env_cfg.vehicle.kinematics.omega_max
        v_max = self.env_cfg.vehicle.kinematics.v_max
        v_min = self.env_cfg.vehicle.kinematics.v_min

        omega = (action[0] + 1) * (0.5*(omega_max - omega_min)) + omega_min
        v = (action[1] + 1) * (0.5*(v_max - v_min)) + v_min
        cmd_vel = np.array([omega, v])

        return cmd_vel
    
    def action(self, act):
        cmd_vel = self.map_normalized_action_to_cmd_vel(act)
    
        return cmd_vel # = action