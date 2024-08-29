import numpy as np
import gymnasium as gym

import utils.admin_tools as at
import matplotlib.pyplot as plt

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

class SparseLidarObservationWrapper(gym.ObservationWrapper):
    params_file_name = 'sparse_lidar_observation.yaml'

    def __init__(self, env):
        super().__init__(env)
        
        # Load parameters
        self.params = at.load_parameters([
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
        # NOTE - the lidar data contains an offset wrt the robot's position!!!
        lidar_data_array = np.array(lidar_data)

        # Nomalize lidar data
        min_range = float(self.params['proto_substitutions']['minRange'])
        max_range = float(self.params['proto_substitutions']['maxRange'])
        normalized_array = normalize(lidar_data_array, min_range, max_range)
        
        # Replace nans and infs with a replacement value
        normalized_array[np.isinf(normalized_array)] = replace_value
        normalized_array[np.isnan(normalized_array)] = replace_value

        return normalized_array

    def process_local_goal(self, local_goal):
        # Convert local goal to local polar coordinates
        goal_pos = np.array(local_goal)
        goal_pos_angle = np.arctan2(goal_pos[1], goal_pos[0])
        goal_pos_dist = np.linalg.norm(goal_pos)
        
        # Clip then normalize goal position
        goal_pos_angle_min = self.params['goal_pos_angle_min']
        goal_pos_angle_max = self.params['goal_pos_angle_max']
        goal_pos_dist_min=self.params['goal_pos_dist_min']
        goal_pos_dist_max=self.params['goal_pos_dist_max']

        clipped_goal_pos_angle = np.clip(goal_pos_angle, goal_pos_angle_min, goal_pos_angle_max)
        clipped_goal_pos_dist = np.clip(goal_pos_dist, goal_pos_dist_min, goal_pos_dist_max)
        if clipped_goal_pos_angle != goal_pos_angle or clipped_goal_pos_dist != goal_pos_dist:
            print("Warning: Goal position has been clipped.")

        goal_pos_angle_normalized = normalize(clipped_goal_pos_angle, goal_pos_angle_min, goal_pos_angle_max)
        goal_pos_dist_normalized = normalize(clipped_goal_pos_dist, goal_pos_dist_min, goal_pos_dist_max)

        return np.array([goal_pos_angle_normalized, goal_pos_dist_normalized])
    
    def process_prev_vel(self, prev_vel):
        omega_min=self.params['omega_min']
        omega_max=self.params['omega_max']
        v_min=self.params['v_min']
        v_max=self.params['v_max']

        if self.unwrapped.plot_wrapped_obs == True:
            print(f"Previous velocity: {prev_vel}")

        omega_normalized = normalize(prev_vel[0], omega_min, omega_max)
        v_normalized = normalize(prev_vel[1], v_min, v_max)

        if self.unwrapped.plot_wrapped_obs == True:
            print(f"Normalized previous omega: {omega_normalized}")
            print(f"Normalized previous v: {v_normalized}")

        return np.array([omega_normalized, v_normalized])

    def observation(self, obs):
        normalized_lidar_data = self.process_lidar_data(
            lidar_data=obs
        )
        normalized_local_goal = self.process_local_goal(
            local_goal=self.unwrapped.local_goal_pos,
        )
        normalized_prev_vel = self.process_prev_vel(
            prev_vel=self.unwrapped.cur_vel
        )

        # Plot observation created by the wrapper to verify correctness
        if self.unwrapped.plot_wrapped_obs == True:
            self.plot_observation(
                normalized_lidar_data, 
                normalized_local_goal, 
                normalized_prev_vel
            )

        return np.concatenate([
            normalized_lidar_data,
            normalized_local_goal,
            normalized_prev_vel
        ])
    
    def init_plot(self):
        plt.ion()
        self.fig9, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.ax1.set_xlim(-1, 1)  # Set the x-axis limits to -1 and 1
        self.ax1.set_ylim(-1, 1)  # Set the y-axis limits to -1 and 1
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_aspect('equal')
        self.ax1.grid(True)
        self.ax1.set_title('Normalized Lidar and Goal Position')

        self.ax2.set_xlim(0, 1)  # Set the x-axis limits to -1 and 1
        self.ax2.set_ylim(0, 1)  # Set the y-axis limits to -1 and 1
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_aspect('equal')
        self.ax2.grid(True)
        self.ax2.set_title('Normalized Previous Velocity')

        self.lidar_plot = None
        self.goal_plot = None
        self.action_plot = None

    def plot_observation(self, normalized_lidar_data, normalized_local_goal, normalized_prev_vel):
        self.clear_plots()
        
        # Convert lidar observations from polar to Cartesian coordinates and plot
        angles = np.linspace(0, 2 * np.pi, len(normalized_lidar_data))
        x_obs = normalized_lidar_data * -np.sin(angles) # NOTE: minus to account for fixed lidar (network should shouldn't care about orientation since it only sees the range data)
        y_obs = normalized_lidar_data * -np.cos(angles)
        self.lidar_plot = self.ax1.scatter(x_obs, y_obs, c='blue', label='Lidar Observations')

        # Convert goal position from polar to Cartesian coordinates and plot
        goal_angle = -np.pi +normalized_local_goal[0]*2*np.pi
        goal_distance = normalized_local_goal[1]
        x_goal = goal_distance * np.cos(goal_angle)
        y_goal = goal_distance * np.sin(goal_angle)
        self.goal_plot = self.ax1.scatter(x_goal, y_goal, c='purple', label='Goal Position')

        # Plot previous action
        goal_pos_angle_normalized = normalized_prev_vel[0]
        goal_pos_dist_normalized = normalized_prev_vel[1]
        self.action_plot = self.ax2.scatter(goal_pos_angle_normalized, goal_pos_dist_normalized, c='green', label='Previous Action')

        self.ax1.legend()
        self.ax2.legend()
        self.ax2.legend()
        self.fig9.canvas.draw()
        plt.pause(0.001)

    def clear_plots(self):
        try:
            self.lidar_plot.remove()
        except Exception:
            pass
        try:
            self.goal_plot.remove()
        except Exception:
            pass
        try:
            self.action_plot.remove()
        except Exception:
            pass