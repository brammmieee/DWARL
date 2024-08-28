import numpy as np
import gymnasium as gym

import utils.admin_tools as at

def map_normalized_action_to_cmd_vel(action, params):
    omega_min = params['omega_min']
    omega_max = params['omega_max']
    v_max = params['v_max']
    v_min = params['v_min']

    omega = (action[0] + 1) * (0.5*(omega_max - omega_min)) + omega_min
    v = (action[1] + 1) * (0.5*(v_max - v_min)) + v_min
    cmd_vel = np.array([omega, v])

    return cmd_vel

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

        # Define action space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    def action(self, act):
        cmd_vel = map_normalized_action_to_cmd_vel(act, self.params)
        cmd_vel = apply_kinematic_constraints(self.params, self.unwrapped.cur_vel, cmd_vel)
    
        return cmd_vel # = action