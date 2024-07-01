import numpy as np
import gymnasium as gym

import utils.admin_tools as at

class ParameterizedRewardWrapper(gym.RewardWrapper):
    '''
    This enviroment wrapper does the following:
    - Converts (by adding to BaseEnv's 0 reward) the reward function to the one descripted in the paper.
    
    '''
    params_file_name = 'parameterized_reward.yaml'

    def __init__(self, BaseEnv):
        super().__init__(BaseEnv)

        self.params = at.load_parameters(['base_parameters.yaml', self.params_file_name])

    def reward(self, r):
        # Adds the reward to the BaseEnv's 0 reward
        
        # Checking if @ goal with low velocity #NOTE: also used to terminate episode
        arrived_at_goal = False
        if (np.linalg.norm(self.unwrapped.cur_pos[:2] - 
                           self.unwrapped.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']): #TODO: pass from DONE instead of recompute
            arrived_at_goal = True #NOTE: arrived with low speed!!!

        # Goal/Progress reward
        if arrived_at_goal:
            r_goal = self.params['c_at_goal']
        else:
            r_goal = self.params['c_progr']*(
                np.linalg.norm(self.unwrapped.goal_pose[:2] - self.unwrapped.prev_pos[:2]) - 
                np.linalg.norm(self.unwrapped.goal_pose[:2] - self.unwrapped.cur_pos[:2])
            )

        # Time penalty
        if arrived_at_goal == False:
            r_speed = -1
        else:
            r_speed = 0

        # Total reward
        reward = r_goal + self.params['c_speed']*r_speed
        
        return reward