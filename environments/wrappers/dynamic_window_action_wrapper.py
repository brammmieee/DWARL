import numpy as np
import gymnasium as gym

import utils.admin_tools as at

class DynamicWindowActionWrapper(gym.ActionWrapper):
    '''
    This enviroment wrapper does the following:
    - Projects the action output of the policy into the dynamic velocity window.
    - Clips the action to the maximal velocity bounds.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.params = at.load_parameters("base_parameters.yaml")

        # Define action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
    
    def action(self, act):
        # Convert the 2D action vector to the dynamic window
        cmd_vel = self.get_action_projection(act)
        
        return cmd_vel # = action
        
    def get_action_projection(self, action):
        # Project 2D action vector inside the dynamic window
        action_projection = np.array([
            self.unwrapped.cur_vel[0] + action[0]*self.params['alpha_max']*self.params['sample_time'],
            self.unwrapped.cur_vel[1] + action[1]*self.params['a_max']*self.params['sample_time'],
        ])

        # Clip action to the maximal velocity bounds
        low=np.array([self.params['omega_min'], self.params['v_min']])
        high=np.array([self.params['omega_max'], self.params['v_max']])
        action_projection = np.clip(a=action_projection, a_min=low, a_max=high)

        return action_projection