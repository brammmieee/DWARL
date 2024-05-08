import numpy as np
import gymnasium as gym

import utils.admin_tools as at

class SparseLidarObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.params = at.load_parameters([ #TODO: list can be directly parsed to init of base env
            "base_parameters.yaml", 
            "sparse_lidar_proto_config.json",
            "sparse_lidar_observation.yaml"
        ])
        
        # Observation space definition
        num_lidar_rays = self.params['proto_substitutions']['horizontalResolution']
        low_array = np.concatenate([
            np.array([float(self.params['proto_substitutions']['minRange'])]*num_lidar_rays),
            np.array([
            float(self.params['goal_pos_dist_min']),
            float(self.params['goal_pos_angle_min']),
            float(self.params['v_min'])
            ])
        ])
        high_array = np.concatenate([
            np.array([float(self.params['proto_substitutions']['maxRange'])]*num_lidar_rays),
            np.array([
            float(self.params['goal_pos_dist_max']),
            float(self.params['goal_pos_angle_max']),
            float(self.params['v_max'])
            ])
        ])
        self.observation_space = gym.spaces.Box(
            low=low_array,
            high=high_array,
            shape=np.shape(low_array),
            dtype=np.float64
        )

    # def get_obs(self):
    #     np.array(self.unwrapped.lidar_range_image)