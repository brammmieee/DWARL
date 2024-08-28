import numpy as np
import gymnasium as gym

import utils.admin_tools as at
import matplotlib.pyplot as plt

class SparseLidarObservationWrapper(gym.ObservationWrapper):
    params_file_name = 'sparse_lidar_observation.yaml'

    def __init__(self, env):
        super().__init__(env)
        
        # Load parameters
        self.params = at.load_parameters([ #TODO: list can be directly parsed to init of base env
            'base_parameters.yaml', 
            'sparse_lidar_proto_config.json',
            self.params_file_name
        ])
        
        # Observation space definition
        num_lidar_rays = self.params['proto_substitutions']['horizontalResolution']
        low_array = np.concatenate([
            np.zeros(num_lidar_rays), #NOTE: since the lidar range image is normalized
            np.zeros(4)
        ])
        high_array = np.concatenate([
            np.ones(num_lidar_rays), #NOTE: since the lidar range image is normalized
            np.ones(4)
        ])
        self.observation_space = gym.spaces.Box(
            low=low_array,
            high=high_array,
            shape=np.shape(low_array),
            dtype=np.float64
        )

        # Some checks
        if self.params['goal_pos_dist_max'] != self.params['proto_substitutions']['maxRange'] \
            or self.params['goal_pos_dist_min'] != self.params['proto_substitutions']['minRange']:
            print("Warning: The max/min goal distance is not equal to the max/min range of the lidar sensor. This may decrease consistency in the observation space.")
        
        # Plotting
        if self.unwrapped.plot_wrapped_obs == True:
            self.init_plot()

    def process_lidar_data(self, lidar_data, replace_value=0):
        lidar_data_array = np.array(lidar_data)

        # Nomalize lidar data
        min_range = float(self.params['proto_substitutions']['minRange'])
        max_range = float(self.params['proto_substitutions']['maxRange'])
        normalized_array = (lidar_data_array - min_range) / (max_range - min_range)
        
        # Replace nans and infs with a replacement value
        normalized_array[np.isinf(normalized_array)] = replace_value
        normalized_array[np.isnan(normalized_array)] = replace_value

    def process_local_goal(self, local_goal, agent_pos):
        # Convert local goal to local polar coordinates
        goal_pos = np.array(local_goal) + np.array(agent_pos)
        goal_pos_dist = np.linalg.norm(goal_pos)
        goal_pos_angle = np.arctan2(goal_pos[1], goal_pos[0])

        # Normalize goal position
        goal_pos_dist_min=self.params['goal_pos_dist_min'],
        goal_pos_dist_max=self.params['goal_pos_dist_max']
        if goal_pos_dist > goal_pos_dist_max:
            print("Warning: Goal position is outside the maximum range. Goal is capped at the maximum range.")
            goal_pos_dist = goal_pos_dist_max
        if goal_pos_dist < goal_pos_dist_min:
            print("Warning: Goal position is outside the minimum range. Goal is capped at the minimum range.")
            goal_pos_dist = goal_pos_dist_min

        goal_pos_dist_normalized = (goal_pos_dist - goal_pos_dist_min) / (goal_pos_dist_max - goal_pos_dist_min)
        goal_pos_angle_normalized = (goal_pos_angle - goal_pos_dist_min) / (goal_pos_dist_max - goal_pos_dist_min)
        
        return np.array([goal_pos_dist_normalized, goal_pos_angle_normalized])
    
    def process_prev_action(self, prev_action):

        # Normalize previous action
        min_range=self.params['v_min'],
        max_range=self.params['v_max']

    def observation(self, obs):
        normalized_lidar_data = self.process_lidar_data(
            lidar_data=obs
        )
        normalized_local_goal = self.process_local_goal(
            local_goal=self.unwrapped.local_goal_pos,
            agent_pos=self.unwrapped.cur_pos
        )
        normalized_prev_action = self.process_prev_action(
            prev_action=self.unwrapped.cur_vel
        )

        # Plot observation created by the wrapper to verify correctness
        if self.unwrapped.plot_wrapped_obs == True:
            self.plot_observation(
                normalized_lidar_data, 
                normalized_local_goal, 
                normalized_prev_action
            )

        return np.concatenate([
            normalized_lidar_data,
            normalized_local_goal,
            normalized_prev_action
        ])

    def init_plot(self):
        plt.ion()
        self.fig9, self.ax9 = plt.subplots()
        self.ax9.set_xlim(0, 1)  # Set the x-axis limits to 0 and 1
        self.ax9.set_ylim(0, 1)  # Set the y-axis limits to 0 and 1
        self.ax9.set_xlabel('X')
        self.ax9.set_ylabel('Y')
        self.ax9.set_title('Normalized Observation')
        self.lidar_plot = None
        self.goal_plot = None

    def plot_observation(self, normalized_lidar_data, normalized_local_goal, normalized_prev_action):
        if self.lidar_plot is not None:
            self.lidar_plot.remove()

        if self.goal_plot is not None:
            self.goal_plot.remove()

        if self.action_plot is not None:
            self.action_plot.remove()

        # Convert lidar observations from polar to Cartesian coordinates and plot
        angles = np.linspace(0, 2 * np.pi, len(normalized_lidar_data))
        x_obs = normalized_lidar_data * np.cos(angles)
        y_obs = normalized_lidar_data * np.sin(angles)
        self.lidar_plot = self.ax9.scatter(x_obs, y_obs, c='b', label='Lidar Observations')

        # Convert goal position from polar to Cartesian coordinates and plot 
        goal_distance = normalized_local_goal[0]
        goal_angle = normalized_local_goal[1]
        x_goal = goal_distance * np.cos(goal_angle)
        y_goal = goal_distance * np.sin(goal_angle)
        self.goal_plot = self.ax9.scatter(x_goal, y_goal, c='r', label='Goal Position')

        # Plot previous action
        self.action_plot = self.ax9.scatter(normalized_prev_action[0], normalized_prev_action[1], c='g', label='Previous Action')


        self.ax9.legend()
        self.fig9.canvas.draw()
        plt.pause(0.001)


        