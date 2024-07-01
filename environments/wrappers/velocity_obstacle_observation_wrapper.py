import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon

import utils.base_tools as bt
import utils.admin_tools as at
import utils.wrappers.velocity_obstacle_tools as ovt

class VelocityObstacleObservationWrapper(gym.ObservationWrapper):
    '''
    This enviroment wrapper does the following:
    - Converts the lidar range image observation from BaseEnv to the observation described in the paper.
    - Adjust the rendering of the observation accordingly.
    - Ads a stuck monitor based on the available admissible velocity space 
      (i.e. the inverse of the calculated velocity obstacle)
    - Bounds the action to the admiss
    
    '''
    params_file_name = "obstacle_velocity_observation.yaml"

    def __init__(self, env):
        super().__init__(env)
        # Load wrapper specific and general parameters
        self.params = at.load_parameters(["base_parameters.yaml", self.params_file_name]) #TODO: list can be directly parsed to init of base env
        
        # Precomputations
        self.precomputed = ovt.precomputations(self.params, visualize=False)
        
        # velocity obstacle observation space
        self.observation_space = gym.spaces.Dict({
            "vel_obs": gym.spaces.Box(
                low=np.array([(self.precomputed['omega_window_min'], self.precomputed['v_window_min'])]*self.params['num_radii']),
                high=np.array([(self.precomputed['omega_window_max'], self.precomputed['v_window_max'])]*self.params['num_radii']),
                shape=(self.params['num_radii'], 2),
                dtype=np.float64
            ),
            "cur_vel": gym.spaces.Box(
                low=np.array([self.params['omega_min'], self.params['v_min']]),
                high=np.array([self.params['omega_max'], self.params['v_max']]),
                shape=(2,),
                dtype=np.float64
            ),
            "dyn_win": gym.spaces.Box(
                low=np.array([(self.params['omega_min'], self.params['v_min'])]*4),
                high=np.array([(self.params['omega_max'], self.params['v_max'])]*4),
                shape=(4, 2),
                dtype=np.float64
            ),
            "goal_vel": gym.spaces.Box(
                low=np.array([self.precomputed['omega_window_min'], self.precomputed['v_window_min']]),
                high=np.array([self.precomputed['omega_window_max'], self.precomputed['v_window_max']]),
                shape=(2,),
                dtype=np.float64
            ),
        })
        
    def observation(self, obs):
        # Converting the lidar_range_image to the lidar_points pointcloud
        self.lidar_points = bt.lidar_to_point_cloud(
            self.params, self.unwrapped.precomputed_lidar_values, obs)

        # Converting lidar_points set in base env's get_obs to the velocity based observation
        self.vel_obs, self.vel_obs_mid = ovt.compute_velocity_obstacle(self.params, self.lidar_points, self.precomputed)
        self.dyn_win = ovt.compute_dynamic_window(self.params, self.unwrapped.cur_vel)
        self.goal_vel = ovt.compute_goal_vel_obs(self.params, self.unwrapped.local_goal_pos, self.unwrapped.cur_vel)
        
        self.obs = {"vel_obs": self.vel_obs_mid, "cur_vel": self.unwrapped.cur_vel, "dyn_win": self.dyn_win, "goal_vel": self.goal_vel}
        self.render()

        return self.obs
    
    def render(self):
        method = self.unwrapped.method
        render_mode = self.unwrapped.render_mode        
        if (render_mode == None) or (render_mode not in ['velocity', 'full']):
            return
        
        # Create initial plot or remove data to prepare for new data
        if self.unwrapped.render_count == 1:
            self.render_init_plot()
        else:
            self.render_remove_data(method)
            
        # Add new data to the plot
        self.render_add_data(method)
        
        # Plot graphs set flags and counter
        self.unwrapped.fig.canvas.draw()
        self.unwrapped.fig.canvas.flush_events()
        self.unwrapped.render_init = False
        self.unwrapped.render_count += 1 # counter for reduced rendering (e.g. every 2nd step)
    
    def render_init_plot(self):
        self.fig_wrap, self.ax_wrap = plt.subplots()
        self.ax_wrap.plot([self.params['omega_max'], self.params['omega_max'], self.params['omega_min'], self.params['omega_min']],
                                [self.params['v_min'], self.params['v_max'], self.params['v_max'], self.params['v_min']], c='blue')
        self.ax_wrap.set_xlim([self.precomputed['omega_window_min'],self.precomputed['omega_window_max']])
        self.ax_wrap.set_ylim([self.precomputed['v_window_min'],self.precomputed['v_window_max']])
        self.ax_wrap.set_xlabel('$\omega$ [rad/s]')
        self.ax_wrap.set_ylabel('v [m/s]')
        self.ax_wrap.set_aspect('equal')
        self.ax_wrap.grid()

    def render_remove_data(self, method):
        try:
            # ax - Clear observation, action and current velocity
            if self.vel_obs_mid.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
                self.vel_obs_mid_plot.remove()
            self.dyn_window_plot.remove()
            self.goal_vel_plot.remove()
            # for list_item in self.goal_vel_line_plot:
            #     list_item.remove()
            # self.vel_obs_patch_plot.remove()
            self.dyn_win_patch_plot.remove()
            self.cur_vel_plot.remove()
            if method == 'step' and self.unwrapped.render_count >= 2:
                self.cmd_vel_plot.remove()
        except AttributeError:
            pass

    def render_add_data(self, method):
        # ax - Observation, action and current velocity
        if self.vel_obs_mid.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
            self.vel_obs_mid_plot = self.ax_wrap.scatter(self.vel_obs_mid[:,0], self.vel_obs_mid[:,1], c='black')
        self.dyn_window_plot = self.ax_wrap.scatter(self.dyn_win[:,0], self.dyn_win[:,1], c='blue')
        self.goal_vel_plot = self.ax_wrap.scatter(self.obs['goal_vel'][0], self.obs['goal_vel'][1], c='purple')
        # self.goal_vel_line_plot = self.ax_wrap.plot([0.0, self.goal_vel_ub[0]], [0.0, self.goal_vel_ub[1]], c='purple')
        # if self.vel_obs.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
        #     vel_obs_patch = plt_polygon(self.vel_obs, alpha=0.17, closed=True, facecolor='black')
        dyn_win_patch = plt_polygon(self.dyn_win, alpha=0.17, closed=True, facecolor='blue')
        # if self.vel_obs.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
        #     self.vel_obs_patch_plot = self.ax_wrap.add_patch(vel_obs_patch)
        self.dyn_win_patch_plot = self.ax_wrap.add_patch(dyn_win_patch)
        self.cur_vel_plot = self.ax_wrap.scatter(self.unwrapped.cur_vel[0], self.unwrapped.cur_vel[1], c='blue', marker='+', alpha=0.99)
        if method == 'step':
            self.cmd_vel_plot = self.ax_wrap.scatter(self.unwrapped.unwrapped.cmd_vel[0], self.unwrapped.unwrapped.cmd_vel[1], color='green', alpha=0.99)