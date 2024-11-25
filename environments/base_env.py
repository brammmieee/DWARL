from environments.webots_env import WebotsEnv
from matplotlib.patches import Polygon as plt_polygon
from shapely.geometry import Point, Polygon
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import utils.env_tools as et

class BaseEnv(gym.Env):
    def __init__(self, cfg, paths, sim_cfg, data_loader, render=False, env_idx=None):
        super().__init__()

        self.cfg = cfg
        self.paths = paths
        self.sim_env = WebotsEnv(sim_cfg, paths)
        self.data_loader = data_loader
        self.env_idx = env_idx
        
        # Simulation environment properties
        self.lidar_resolution = self.sim_env.lidar_resolution
        self.sample_time = self.sim_env.sample_time
        
        # Precomputations
        self.lidar_precomputation = et.lidar_precomputation(num_lidar_rays=self.lidar_resolution)
        
        # Rendering
        self.render_ = render
        self.render_init_plot()
        
        # Space definitions
        # NOTE: u must use an action wrapper to set self.action_space
        # NOTE: u must use an observation wrapper to set self.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None)
        
        # Reset map, path, init/goal pose, simulation and collision tree
        self.current_data = self.data_loader.get_data_for_env(self.env_idx)
        self.map_name, self.map, self.path, self.init_pose, self.goal_pose = self.current_data.values()
            
        # Resetting the simulation
        self.sim_env.reset()
        self.sim_env.reset_map(self.map_name)
        self.sim_env.reset_robot(self.init_pose)
        
        # Precomputations
        self.collision_tree = et.compute_collision_detection_tree(self.map)
        self.map_bounds_polygon = et.compute_map_bound_polygon(self.map, self.cfg.map_padding)
       
        # Resetting the environment state
        self.cur_pos = self.sim_env.robot_position
        self.cur_orient_matrix = self.sim_env.robot_orientation
        self.cur_vel, self.cmd_vel = np.array([0.0, 0.0]), np.array([0.0, 0.0])
        self.observation, _, done, _, info = self.step(self.cmd_vel)
        
        # Check if the initial pose is valid, if not reset the environment again
        if done:
            self.reset(seed=seed, options=options)

        # Rendering
        self.render(method='reset')

        return self.observation, info

    def step(self, action):
        # NOTE: u must use an action wrapper for turning the action vector into a cmd_vel
        self.cmd_vel = action
           
        # Inacting the action (i.e. limit velocity to kinematically feasible values and forward simulate robot)
        self.cur_vel = et.apply_kinematic_constraints(self.sample_time, self.cfg.vehicle.kinematics, self.cur_vel, self.cmd_vel)
        pos, orient = et.compute_new_pose(self.sample_time, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        self.sim_env.step(pos, orient)

        # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        self.prev_pos = self.cur_pos
        self.cur_pos = self.sim_env.robot_position
        self.cur_orient_matrix = self.sim_env.robot_orientation
        self.footprint_glob = et.get_global_footprint_location(
            self.cur_pos, 
            self.cur_orient_matrix, 
            self.cfg.vehicle.dimensions.polygon_coordinates
        )
        self.local_goal_pos = et.get_local_goal_pos(
            self.cur_pos,
            self.cur_orient_matrix, 
            self.goal_pose
        )
        # Getting new observation, done, reward
        self.observation = self.get_obs()
        self.done, self.done_cause = self.get_done()        
        self.reward = self.get_reward()
        
        # Rendering
        self.render(method='step')

        return self.observation, self.reward, self.done, False, {} # last 3: done, truncated, info
    
    def get_reward(self):
        # NOTE: u must use a reward wrapper
        return 0.0

    def get_obs(self):
        # super().step(self.basic_timestep) # NOTE: moved to sim_env but might be needed here (weird bugfix)
        self.lidar_range_image = self.sim_env.lidar_range_image

        return self.lidar_range_image
        
    def get_done(self):
        # Define conditions and their corresponding done causes
        conditions = {
            'at_goal': np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.cfg.goal_tolerance,
            # 'outside_map': not (self.map_bounds_polygon.contains(Point(self.cur_pos[0], self.cur_pos[1])) or 
            #                     self.map_bounds_polygon.boundary.contains(Point(self.cur_pos[0], self.cur_pos[1]))),
            'collision': len(self.collision_tree.query(self.footprint_glob, predicate='intersects')) > 0
        }

        # Check for done conditions
        for cause, condition in conditions.items():
            if condition:
                return True, cause

        # When none of the done conditions are met
        return False, None
    
    def close(self):
        self.sim_env.close()

    def render(self, method=None):
        if not self.render_:
            return
        
        # Create initial plot or remove data
        self.render_remove_data(method)
        self.render_add_data(method)
        
        # Plot graphs set flags and counter
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def render_init_plot(self):
        if not self.render_:
            return
        
        # Initialize the plot   
        plt.ion()
        self.fig, (self.ax, self.ax1) = plt.subplots(1, 2)
        
        # ax - Map, path, robot, goal, and local goal (axis fixed to map)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        
        # ax1 - Lidar data and footprint (axis fixed to base_link)
        polygon = Polygon(self.cfg.vehicle.dimensions.polygon_coordinates)
        patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.75, closed=True, facecolor='grey')
        self.ax1.add_patch(patch)
        self.ax1.set_xlim([-3.0, 3.0])
        self.ax1.set_ylim([-3.0, 3.0])
        self.ax1.set_xlabel('x [m]')
        self.ax1.set_ylabel('y [m]')
        self.ax1.set_aspect('equal')
        self.ax1.grid()
        
    def render_add_data(self, method):
        # ax - Map, path, robot, goal, and local goal (axis fixed to map)
        self.cur_pos_plot = self.ax.scatter(self.cur_pos[0], self.cur_pos[1], c='blue', alpha=0.33)
        x, y = self.footprint_glob.exterior.xy
        self.footprint_plot = self.ax.fill(x, y, color='blue', alpha=0.5)
        if method == 'reset':
            self.map_plots = []  # Initialize a list to store all rectangle patches
            for box in self.map:
                min_x = min(vertex[0] for vertex in box)
                min_y = min(vertex[1] for vertex in box)
                width = max(vertex[0] for vertex in box) - min_x
                height = max(vertex[1] for vertex in box) - min_y
                rect = plt.Rectangle((min_x, min_y), width, height, color='black')
                self.map_plots.append(self.ax.add_patch(rect))  # Add each patch to the list
        
            self.path_plot = self.ax.scatter(self.path[:,0], self.path[:,1], c='grey', alpha=0.5)
            self.init_pose_plot = self.ax.scatter(self.init_pose[0], self.init_pose[1], c='green')
            self.goal_pose_plot = self.ax.scatter(self.goal_pose[0], self.goal_pose[1], c='red')
        
        # ax1 - Lidar data
        self.lidar_points = et.lidar_to_point_cloud(self.cfg.vehicle.dimensions.lidar_y_offset, self.lidar_precomputation, self.lidar_range_image) #NOTE: double computation in case of e.g vo observation wrapper
        self.lidar_plot = self.ax1.scatter(self.lidar_points[:,0], self.lidar_points[:,1], alpha=1.0, c='black')
        self.local_goal_plot = self.ax1.scatter(self.local_goal_pos[0], self.local_goal_pos[1], alpha=1.0, c='purple')               
    
    def render_remove_data(self, method):
        # ax - Map, path, robot, goal, and local goal (axis fixed to map)
        try:
            self.cur_pos_plot.remove()
            for patch in self.footprint_plot:
                patch.remove()
            if method == 'reset':
                for patch in self.map_plots:
                    patch.remove()
                self.map_plots.clear()
                self.path_plot.remove()
                self.init_pose_plot.remove()
                self.goal_pose_plot.remove()
        except AttributeError:
            pass
        
        # ax1 - Lidar data
        try:
            self.lidar_plot.remove()
            self.local_goal_plot.remove()
        except AttributeError:
            pass