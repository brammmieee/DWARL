import numpy as np
import gymnasium as gym

import utils.admin_tools as at

def normalize_lidar_range_image(lidar_range_image, min_range, max_range):
    '''
    Normalizes the lidar range image to be between 0 and 1 and convert it from a list to a numpy array.
    '''
    lidar_range_image_array = np.array(lidar_range_image)
    min_range = float(min_range)
    max_range = float(max_range)
    normalized_array = (lidar_range_image_array - min_range) / (max_range - min_range)

    return normalized_array

def remove_infinite_values(lidar_range_image_array, replace_value):
    '''
    Removes infinite values from the lidar range image array and replaces them with a specified value.
    '''
    lidar_range_image_array[np.isinf(lidar_range_image_array)] = replace_value
    return lidar_range_image_array

def convert_local_goal_to_polar_coords(local_goal, agent_pos):
    '''
    Converts the local goal to polar coordinates with respect to the agent position.
    '''
    goal_pos = np.array(local_goal) + np.array(agent_pos)
    goal_pos_dist = np.linalg.norm(goal_pos)
    goal_pos_angle = np.arctan2(goal_pos[1], goal_pos[0])

    return goal_pos_dist, goal_pos_angle

class SparseLidarObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.params = at.load_parameters([ #TODO: list can be directly parsed to init of base env
            'base_parameters.yaml', 
            'sparse_lidar_proto_config.json',
            'sparse_lidar_observation.yaml'
        ])
        
        # Observation space definition
        num_lidar_rays = self.params['proto_substitutions']['horizontalResolution']
        low_array = np.concatenate([
            np.array([0]*num_lidar_rays), #NOTE: since the lidar range image is normalized
            np.array([
                float(self.params['goal_pos_dist_min']),
                float(self.params['goal_pos_angle_min']),
                float(self.params['omega_min']),
                float(self.params['v_min'])
            ])
        ])
        high_array = np.concatenate([
            np.array([1]*num_lidar_rays), #NOTE: since the lidar range image is normalized
            np.array([
                float(self.params['goal_pos_dist_max']),
                float(self.params['goal_pos_angle_max']),
                float(self.params['omega_max']),
                float(self.params['v_max'])
            ])
        ])
        self.observation_space = gym.spaces.Box(
            low=low_array,
            high=high_array,
            shape=np.shape(low_array),
            dtype=np.float64
        )

    def observation(self, obs):
        lidar_range_image_array = normalize_lidar_range_image(
            lidar_range_image=obs,
            min_range=self.params['proto_substitutions']['minRange'],
            max_range=self.params['proto_substitutions']['maxRange'],
        )
        lidar_range_image_array = remove_infinite_values( 
            lidar_range_image_array=lidar_range_image_array,
            replace_value=1 #NOTE: could also be set to 0 instead! (1 is max normalized range)
        )
        goal_pos_dist, goal_pos_angle = convert_local_goal_to_polar_coords(
            local_goal=self.unwrapped.local_goal_pos,
            agent_pos=self.unwrapped.cur_pos
        )
        self.obs = np.concatenate([
            lidar_range_image_array,
            np.array([
                goal_pos_dist, goal_pos_angle, 
                self.unwrapped.cur_vel[0], 
                self.unwrapped.cur_vel[1]
            ])
        ])

        return self.obs


        