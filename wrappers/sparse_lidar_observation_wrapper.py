import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

class SparseLidarObservation(gym.ObservationWrapper):
    params_file_name = 'sparse_lidar_observation.yaml'

    def __init__(self, env, cfg):
        super().__init__(env)
        
        self.cfg = cfg
        self.kinematic_cfg = self.unwrapped.cfg.vehicle.kinematics
        
        # Observation space definition
        num_lidar_rays = self.unwrapped.sim_env.lidar_resolution
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
        if self.cfg.goal_pos_dist_max != self.unwrapped.sim_env.lidar_max_range \
            or self.cfg.goal_pos_dist_min != self.unwrapped.sim_env.lidar_min_range:
            print("Warning: The max/min goal distance is not equal to the max/min range of the lidar sensor. This may decrease consistency in the observation space.")
        
        # Plotting
        if self.cfg.render == True:
            self.init_plot()

        # Footprint distances for removing footprint from lidar data
        self.footprint_polygon = self.unwrapped.cfg.vehicle.dimensions.polygon_coordinates
        self.footprint_distances = self.calculate_footprint_distances()
    
    def calculate_footprint_distances(self):
        footprint = np.array(self.footprint_polygon)
        lidar_offset = np.array([0, self.unwrapped.cfg.vehicle.dimensions.lidar_y_offset])
        
        # Adjust footprint coordinates relative to lidar position
        adjusted_footprint = footprint - lidar_offset
        
        angles = np.linspace(0, 2 * np.pi, self.unwrapped.sim_env.lidar_resolution, endpoint=False)
        
        distances = []
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            # Project adjusted footprint points onto the ray direction
            projections = np.dot(adjusted_footprint, direction)
            # Take maximum projection as the distance
            max_projection = np.max(projections)
            # Ensure non-negative distances
            distances.append(max(0.0, max_projection))
        
        return np.array(distances)

    def process_lidar_data(self, lidar_data):
        # NOTE - the lidar data contains an offset wrt the robot's position!!!

        # Remove footprint from lidar data
        adjusted_lidar_data = np.maximum(lidar_data - self.footprint_distances, 0)

        # Parameters
        min_range = float(self.unwrapped.sim_env.lidar_min_range)
        max_range = float(self.unwrapped.sim_env.lidar_max_range)

        # Clip and replace invalid values with max range
        adjusted_lidar_data[np.isinf(adjusted_lidar_data)] = max_range # this can happen when the lidar sensor doesn't detect anything
        if np.isnan(adjusted_lidar_data).any():
            print("Warning: Lidar data contains NaN values. Replacing with max range.")
        adjusted_lidar_data[np.isnan(adjusted_lidar_data)] = max_range # this shoudn't happen ergo the warning

        # Nomalize lidar data
        normalized_array = normalize(adjusted_lidar_data, min_range, max_range)

        return normalized_array

    def process_local_goal(self, local_goal):
        # Convert local goal to local polar coordinates
        goal_pos = np.array(local_goal)
        goal_pos_angle = np.arctan2(goal_pos[0], goal_pos[1])
        goal_pos_dist = np.linalg.norm(goal_pos)

        # Clip then normalize goal position
        goal_pos_angle_min = self.cfg.goal_pos_angle_min
        goal_pos_angle_max = self.cfg.goal_pos_angle_max
        goal_pos_dist_min=self.cfg.goal_pos_dist_min
        goal_pos_dist_max=self.cfg.goal_pos_dist_max

        clipped_goal_pos_angle = np.clip(goal_pos_angle, goal_pos_angle_min, goal_pos_angle_max)
        clipped_goal_pos_dist = np.clip(goal_pos_dist, goal_pos_dist_min, goal_pos_dist_max)
        if clipped_goal_pos_angle != goal_pos_angle:
            print(f"Warning: goal_pos_angle {goal_pos_angle} has been clipped to {clipped_goal_pos_angle}.")
        if clipped_goal_pos_dist != goal_pos_dist:
            print(f"Warning: goal_pos_dist {goal_pos_dist} has been clipped to {clipped_goal_pos_dist}.")

        goal_pos_angle_normalized = normalize(clipped_goal_pos_angle, goal_pos_angle_min, goal_pos_angle_max)
        goal_pos_dist_normalized = normalize(clipped_goal_pos_dist, goal_pos_dist_min, goal_pos_dist_max)

        return np.array([goal_pos_angle_normalized, goal_pos_dist_normalized])
    
    def process_prev_vel(self, prev_vel):
        omega_min=self.kinematic_cfg.omega_min
        omega_max=self.kinematic_cfg.omega_max
        v_min=self.kinematic_cfg.v_min
        v_max=self.kinematic_cfg.v_max

        omega_normalized = normalize(prev_vel[0], omega_min, omega_max)
        v_normalized = normalize(prev_vel[1], v_min, v_max)

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
        if self.cfg.render == True:
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