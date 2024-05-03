import numpy as np
import gymnasium as gym
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

import utils.admin_tools as at

class DynamicWindowActionWrapper(gym.ActionWrapper):
    '''
    This enviroment wrapper does the following:
    - Projects the action output of the policy into the dynamic velocity window.
    
    '''
    def __init__(self, env):
        super().__init__(env)
        self.params = at.load_parameters("general_parameters.yaml")

        # Define action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
    def action(self, act):
        # Convert the 2D action vector to the dynamic window
        self.cmd_vel = self.get_action_projection(act)
        
        return self.cmd_vel # = action
    
    def get_done(self):
        # Add additional done cause (i.e. getting stuck)
        done = self.env.get_done()
        # Check if the agent is stuck
        if self.stuck:
            return True
        else:
            return done
        
    def get_action_projection(self, action):
        # Project 2D action vector inside the dynamic window
        action_projection = np.array([
            self.cur_vel[0] + action[0]*self.params['alpha_max']*self.params['sample_time'],
            self.cur_vel[1] + action[1]*self.params['a_max']*self.params['sample_time'],
        ])

        # Clip action to the maximal velocity bounds
        low=np.array([self.params['omega_min'], self.params['v_min']])
        high=np.array([self.params['omega_max'], self.params['v_max']])
        action_projection = np.clip(a=action_projection, a_min=low, a_max=high)

        # If action inside vel_obs (and hence outside adm_vel_polygon), take closest safe point instead
        if not np.all(self.vel_obs == 0):
            adm_vel_polygon = Polygon(self.vel_obs)
            action_point = Point(action_projection[0], action_projection[1])
            if not adm_vel_polygon.contains(action_point):
                if not adm_vel_polygon.boundary.contains(action_point):
                    with np.errstate(invalid='ignore'):
                        closest_point, _ = nearest_points(adm_vel_polygon, action_point)
                    action_projection = np.array([closest_point.x, closest_point.y])
        else:
            action_projection = np.array([0.0, 0.0])
            self.stuck = True

        return action_projection