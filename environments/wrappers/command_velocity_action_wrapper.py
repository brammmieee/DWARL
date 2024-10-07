import numpy as np
import gymnasium as gym

import utils.admin_tools as at

class CommandVelocityActionWrapper(gym.ActionWrapper):
    '''
    This enviroment wrapper does the following:
    - Projects the action output of the policy into the dynamic velocity window.
    - Clips the action to the maximal velocity bounds.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.params = at.load_parameters("base_parameters.yaml")

        # Define action space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    def map_normalized_action_to_cmd_vel(self, action, params):
        omega_min = params['omega_min']
        omega_max = params['omega_max']
        v_max = params['v_max']
        v_min = params['v_min']

        omega = (action[0] + 1) * (0.5*(omega_max - omega_min)) + omega_min
        v = (action[1] + 1) * (0.5*(v_max - v_min)) + v_min
        cmd_vel = np.array([omega, v])

        return cmd_vel
    
    def action(self, act):
        cmd_vel = self.map_normalized_action_to_cmd_vel(act, self.params)
    
        return cmd_vel # = action