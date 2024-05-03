import numpy as np
import gymnasium as gym

from environments.base_env import BaseEnv

class PaperRewardWrapper(gym.RewardWrapper):
    '''
    This enviroment wrapper does the following:
    - Converts (by adding to BaseEnv's 0 reward) the reward function to the one descripted in the paper.
    
    '''
    def __init__(self, BaseEnv):
        super().__init__(BaseEnv)
        
    def reset(self, seed=None):
        # Add the 
        self.env.reset(seed=seed)
        
        # Reset reward variables
        self.stuck = False

    def reward(self, r):
        # Adds the reward to the BaseEnv's 0 reward
        
        # Checking if @ goal with low velocity #NOTE: also used to terminate episode
        arrived_at_goal = False
        if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']): #TODO: pass from DONE instead of recompute
            arrived_at_goal = True #NOTE: arrived with low speed!!!

        # Goal/Progress reward
        if arrived_at_goal:
            r_goal = self.params['c_at_goal']
        else:
            r_goal = self.params['c_progr']*(np.linalg.norm(self.goal_pose[:2] - self.prev_pos[:2]) - np.linalg.norm(self.goal_pose[:2] - self.cur_pos[:2]))

        # Time penalty
        if arrived_at_goal == False:
            r_speed = -1
        else:
            r_speed = 0

        # Stuck penalty
        if self.stuck == True:
            r_stuck = -1.0
        else:
            r_stuck = 0.0

        # Total reward
        reward = r_goal + self.params['c_speed']*r_speed + self.params['c_stuck']*r_stuck
        
        return reward