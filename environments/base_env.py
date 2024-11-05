import os
import random
import numpy as np
import math as m
import gymnasium as gym
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon
from shapely.affinity import translate, rotate
from scipy.interpolate import interp1d
from pathlib import Path
from utils.admin_tools import load_data_set_config, convert_point_from_image_base

import utils.env_tools as et

GRID_TRESHOLD = 255

class BaseEnv(gym.Env):
    def __init__(self, cfg, paths, sim_env, data_loader, env_idx, render_mode='something'):
        super().__init__()

        self.cfg = cfg
        self.paths = paths
        self.sim_env = sim_env
        self.data_loader = data_loader
        self.env_idx = env_idx
        
        # Simulation environment properties
        self.lidar_resolution = sim_env.lidar_resolution
        self.sample_time = sim_env.sample_time
        
        # Dataset properties
        path_to_config = Path(self.paths.data_sets.config) / "config.yaml"
        data_set_config = load_data_set_config(path_to_config)
        self.map_cfg = data_set_config.map
                
        # Update the proto files according to the configuration
        # et.update_protos(self.cfg.proto_reconfiguration, self.paths.resources.protos) # NOTE: moved to asset_generator
        
        # Precomputations
        self.lidar_precomputation = et.lidar_precomputation(num_lidar_rays=sim_env.lidar_resolution)
        
        # Rendering
        self.render_mode = render_mode
        self.render_init_plot()
        
        # Space definitions
        # NOTE: u must use an action wrapper to set self.action_space
        # NOTE: u must use an observation wrapper to set self.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None)
        
        # Reset map, path, init/goal pose, simulation and collision tree
        self.current_data = self.data_loader.get_data_for_env(self.env_idx)
        proto_name, self.grid, self.path, self.init_pose, self.goal_pose = self.current_data.values()
            
        # Resetting the simulation
        self.sim_env.reset()
        self.sim_env.reset_map(proto_name)
        self.sim_env.reset_robot(self.init_pose)
        
        self.collision_tree = et.compute_collision_detection_tree(self.grid, self.map_cfg.resolution)
        self.map_bounds_polygon = et.compute_map_bound_polygon(
            res=self.map_cfg.resolution,
            width=self.grid.shape[1],
            height=self.grid.shape[0],
            padding=self.map_cfg.padding
        )

        # # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
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
        self.cur_vel, self.cmd_vel = np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
        self.observation = self.get_obs()
        self.render(method='reset')

        return self.observation, {} # = info

    def step(self, action):
        # NOTE: u must use an action wrapper for turning the action vector into a cmd_vel
        self.cmd_vel = action

        # Updating previous values before updating
        self.prev_pos = self.cur_pos
        self.prev_observation = self.observation
           
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

        self.observation = self.get_obs()
        self.done, self.done_cause = self.get_done()
        self.reward = self.get_reward()
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
        done_cause = None

        # Arrived at the goal
        if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.map_cfg.goal_tolerance): #and (self.cur_vel[1] < self.params['v_goal_threshold']):
            done_cause = 'at_goal'
            return True, done_cause
            
        # Driving outside map limit
        cur_pos_point = Point(self.cur_pos[0], self.cur_pos[1])
        if not (self.map_bounds_polygon.contains(cur_pos_point) or self.map_bounds_polygon.boundary.contains(cur_pos_point)):
            done_cause = 'outside_map'
            return True, done_cause
        
        # Collision with obstacles
        if len(self.collision_tree.query(self.footprint_glob, predicate='intersects')) > 0:
            done_cause = 'collision'
            return True, done_cause

        # When none of the done conditions are met
        return False, done_cause
    
    def close(self):
        self.sim_env.close()

    def render(self, method=None):
        if self.render_mode == None:
            return
        
        # Create initial plot or remove data
        self.render_remove_data(method)
        self.render_add_data(method)
        
        # Plot graphs set flags and counter
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def render_init_plot(self):
        if self.render_mode == 'none':
            return
        
        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
    
    def render_add_data(self, method):
        self.cur_pos_plot = self.ax.scatter(self.cur_pos[0], self.cur_pos[1], c='blue', alpha=0.33)
        x, y = self.footprint_glob.exterior.xy
        self.footprint_plot = self.ax.fill(x, y, color='blue', alpha=0.5)
        if method == 'reset':
            self.grid_plots = []  # Initialize a list to store all rectangle patches
            indices = np.argwhere(self.grid < GRID_TRESHOLD)
            for index in indices:
                x, y = index
                point = np.array([float(x),float(y)])
                print(f"Point: {point}, type: {type(point)}, type[0]: {type(point[0])}")
                converted_point = convert_point_from_image_base(point, self.map_cfg.resolution, self.grid.shape[0])
                rect = plt.Rectangle((converted_point[0], converted_point[1]), self.map_cfg.resolution, self.map_cfg.resolution, color='black')
                self.grid_plots.append(self.ax.add_patch(rect))  # Add each patch to the list
        
            self.path_plot = self.ax.scatter(self.path[:,0], self.path[:,1], c='grey', alpha=0.5)
            self.init_pose_plot = self.ax.scatter(self.init_pose[0], self.init_pose[1], c='green')
            self.goal_pose_plot = self.ax.scatter(self.goal_pose[0], self.goal_pose[1], c='red')
            
            # import ipdb; ipdb.set_trace()
            
    def render_remove_data(self, method):
        try:
            self.cur_pos_plot.remove()
            for patch in self.footprint_plot:
                patch.remove()
            if method == 'reset':
                for patch in self.grid_plots:
                    patch.remove()
                self.grid_plots.clear()
                self.path_plot.remove()
                self.init_pose_plot.remove()
                self.goal_pose_plot.remove()
        except AttributeError:
            pass