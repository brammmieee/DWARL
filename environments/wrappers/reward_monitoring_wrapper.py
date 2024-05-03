

import os
import gymnasium as gym

class RewardMonitoringWrapper(gym.Wrapper):
    '''
    This enviroment wrapper does the following:
    - Converts the lidar range image observation from BaseEnv to the observation described in the paper.
    - Adjust the rendering of the observation accordingly.
    
    '''
    def __init__(self, BaseEnv):
        super().__init__(BaseEnv)
        # Reward monitoring
        self.reward_monitoring = reward_monitoring
        if self.reward_monitoring == True:
            self.episode_nr = 1
            self.reward_matrix = []

    def reset(self, seed=None):
        # Reward monitoring
        if self.reward_monitoring == True:
            self.reward_list.append(self.reward)
            if done:
                self.reward_matrix.append(self.reward_list)
                at.write_pickle_file(f'rewards_ep_{self.episode_nr}', os.path.join('training','rewards'), self.reward_matrix)
                self.episode_nr += 1
    
    def step(self, action, teleop=False):
        # Reset reward list
        if self.reward_monitoring == True:
            self.reward_list = []