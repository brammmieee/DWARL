import numpy as np
import gymnasium as gym

import utils.admin_tools as at
import matplotlib.pyplot as plt

def normalize_lidar_range_image(lidar_range_image, min_range, max_range):
    '''
    Normalizes the lidar range image to be between 0 and 1 and convert it from a list to a numpy array.
    '''
    lidar_range_image_array = np.array(lidar_range_image)
    min_range = float(min_range)
    max_range = float(max_range)
    normalized_array = (lidar_range_image_array - min_range) / (max_range - min_range)

    return normalized_array

def remove_invalid_values(lidar_range_image_array, replace_value):
    '''
    Removes infinite and NaN values from the lidar range image array and replaces them with a specified value.
    
    Parameters:
        lidar_range_image_array (np.ndarray): The array containing lidar range image data.
        replace_value (float): The value to use as a replacement for infinite and NaN values.

    Returns:
        np.ndarray: The modified array with no infinite or NaN values.
    '''
    # Replace infinite values
    lidar_range_image_array[np.isinf(lidar_range_image_array)] = replace_value
    # Replace NaN values
    lidar_range_image_array[np.isnan(lidar_range_image_array)] = replace_value
    return lidar_range_image_array


def convert_local_goal_to_polar_coords(local_goal, agent_pos):
    '''
    Converts the local goal to polar coordinates with respect to the agent position.
    '''
    goal_pos = np.array(local_goal) + np.array(agent_pos)
    goal_pos_dist = np.linalg.norm(goal_pos)
    goal_pos_angle = np.arctan2(goal_pos[1], goal_pos[0])

    return goal_pos_dist, goal_pos_angle

def normalize_goal_position(goal_pos_dist, goal_pos_angle, goal_pos_dist_min, goal_pos_dist_max):
    '''
    Normalizes the goal distance and angle based on the provided parameters.
    '''
    if goal_pos_dist > goal_pos_dist_max:
        print("Warning: Goal position is outside the maximum range. Goal is capped at the maximum range.")
        goal_pos_dist = goal_pos_dist_max
    if goal_pos_dist < goal_pos_dist_min:
        print("Warning: Goal position is outside the minimum range. Goal is capped at the minimum range.")
        goal_pos_dist = goal_pos_dist_min

    goal_pos_dist_normalized = (goal_pos_dist - goal_pos_dist_min) / (goal_pos_dist_max - goal_pos_dist_min)
    goal_pos_angle_normalized = (goal_pos_angle - goal_pos_dist_min) / (goal_pos_dist_max - goal_pos_dist_min)
    return goal_pos_dist_normalized, goal_pos_angle_normalized


class SparseLidarObservationWrapper(gym.ObservationWrapper):
    params_file_name = 'sparse_lidar_observation.yaml'

    def __init__(self, env):
        super().__init__(env)
        
        self.params = at.load_parameters([ #TODO: list can be directly parsed to init of base env
            'base_parameters.yaml', 
            'sparse_lidar_proto_config.json',
            self.params_file_name
        ])
        
        # Observation space definition
        num_lidar_rays = self.params['proto_substitutions']['horizontalResolution']
        low_array = np.concatenate([
            np.array([0]*num_lidar_rays), #NOTE: since the lidar range image is normalized
            np.array([
                float(0),
                float(0),
                float(0),
                float(0)
            ])
        ])
        high_array = np.concatenate([
            np.array([1]*num_lidar_rays), #NOTE: since the lidar range image is normalized
            np.array([
                float(1),
                float(1),
                float(1),
                float(1)
            ])
        ])
        self.observation_space = gym.spaces.Box(
            low=low_array,
            high=high_array,
            shape=np.shape(low_array),
            dtype=np.float64
        )

        if self.params['goal_pos_dist_max'] != self.params['proto_substitutions']['maxRange'] \
            or self.params['goal_pos_dist_min'] != self.params['proto_substitutions']['minRange']:
            print("Warning: The max/min goal distance is not equal to the max/min range of the lidar sensor. This may decrease consistency in the observation space.")

    def observation(self, obs):
        # Lidar range normalization and invalid value removal
        normalized_lidar_range_image_array = normalize_lidar_range_image(
            lidar_range_image=obs,
            min_range=self.params['proto_substitutions']['minRange'],
            max_range=self.params['proto_substitutions']['maxRange'],
        )
        valid_only_normalized_lidar_range_image_array = remove_invalid_values( 
            lidar_range_image_array=normalized_lidar_range_image_array,
            replace_value=0 #NOTE: could also be set to 0 instead! (1 is max normalized range)
        )

        # Goal position normalization
        goal_pos_dist, goal_pos_angle = convert_local_goal_to_polar_coords(
            local_goal=self.unwrapped.local_goal_pos,
            agent_pos=self.unwrapped.cur_pos
        )
        normalized_goal_pos_dist, normalized_goal_pos_angle = normalize_goal_position(
            goal_pos_dist=goal_pos_dist,
            goal_pos_angle=goal_pos_angle,
            goal_pos_dist_min=self.params['goal_pos_dist_min'],
            goal_pos_dist_max=self.params['goal_pos_dist_max']
        )

        # # Previous action normalization
        # prev_action = self.unwrapped.cur_vel
        # self.normalized_prev_action = normalize_prev_action(
        #     prev_action=prev_action,
        #     min_range=self.params['v_min'],
        #     max_range=self.params['v_max']
        # )

        # Observation concatenation
        self.obs = np.concatenate([
            valid_only_normalized_lidar_range_image_array,
            np.array([normalized_goal_pos_dist, normalized_goal_pos_angle])
        ])

        if self.unwrapped.plot_wrapped_state == True:
            self.plot_observation(
                valid_only_normalized_lidar_range_image_array, 
                normalized_goal_pos_dist, 
                normalized_goal_pos_angle
            )

        return self.obs
    
    def plot_observation(self, lidar_range_image, goal_angle, goal_distance):
        try:
            self.fig9 is None
        except Exception:
            self.fig9, self.ax9 = plt.subplots()
            self.ax9.set_xlim(0, 1)  # Set the x-axis limits to 0 and 1
            self.ax9.set_ylim(0, 1)  # Set the y-axis limits to 0 and 1
            self.ax9.set_xlabel('X')
            self.ax9.set_ylabel('Y')
            self.ax9.set_title('Normalized Lidar Range Image')

        self.ax9.clear()

        # Convert lidar observations from polar to Cartesian coordinates
        angles = np.linspace(0, 2 * np.pi, len(lidar_range_image))
        x_obs = lidar_range_image * np.cos(angles)
        y_obs = lidar_range_image * np.sin(angles)

        # Plot lidar observations
        self.ax9.scatter(x_obs, y_obs, c='b', label='Lidar Observations')

        # Convert goal position from polar to Cartesian coordinates
        x_goal = goal_distance * np.cos(goal_angle)
        y_goal = goal_distance * np.sin(goal_angle)

        # Plot goal position
        self.ax9.scatter(x_goal, y_goal, c='r', label='Goal Position')

        self.ax9.legend()
        self.fig9.canvas.draw()
        plt.pause(0.001)



        