import numpy as np
import gymnasium as gym

import utils.admin_tools as at

def apply_kinematic_constraints(params, cur_vel, target_vel):
    omega_max = params['omega_max']
    omega_min = params['omega_min']
    alpha_max = params['alpha_max']
    alpha_min = params['alpha_min']
    a_max = params['a_max']
    a_min = params['a_min']
    v_max = params['v_max']
    v_min = params['v_min']
    dt = 1/params['sample_time']

    domega = target_vel[0] - cur_vel[0]
    domega_clipped = np.clip(domega, alpha_min*dt, alpha_max*dt) 
    omega_clipped = np.clip((cur_vel[0] + domega_clipped), omega_min, omega_max)

    dv = target_vel[1] - cur_vel[1]
    dv_clipped = np.clip(dv, a_min*dt, a_max*dt)
    v = np.clip((cur_vel[1] + dv_clipped), v_min, v_max)

    return np.array([omega_clipped, v])

class CommandVelocityActionWrapper(gym.ActionWrapper):
    '''
    This enviroment wrapper does the following:
    - Projects the action output of the policy into the dynamic velocity window.
    - Clips the action to the maximal velocity bounds.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.params = at.load_parameters("base_parameters.yaml")
        v_max = self.params['v_max']
        v_min = self.params['v_min']
        omega_min = self.params['omega_min']
        omega_max = self.params['omega_max']

        # Define action space
        self.action_space = gym.spaces.Box(
            low=np.array([v_min, omega_min]), 
            high=np.array([v_max, omega_max]),
            shape=(2,), 
            dtype=np.float32
        )
        
    
    def action(self, act):
        # Convert the 2D action vector to the dynamic window
        cmd_vel = apply_kinematic_constraints(self.params, self.unwrapped.cur_vel, act)

        return cmd_vel # = action