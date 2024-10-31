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

import utils.env_tools as et
import utils.admin_tools as at

class BaseEnv(gym.Env):
    package_dir = os.path.abspath(os.pardir)
    
    def __init__(self, cfg, paths, data_set):
        super().__init__()

        self.cfg = cfg
        self.paths = paths
        # self.render_canvas_drawn = False

        # Update the proto files according to the configuration
        # et.update_protos(self.cfg.proto_reconfiguration, self.paths.resources.protos)
        
        # Precomputations
        self.precomputed_lidar_values = et.precompute_lidar_values(
            num_lidar_rays=cfg.proto_reconfiguration.substitutions.horizontalResolution
        )

        # Create a Webots environment
        self.webots_env = et.WebotsEnv(cfg.webots, paths)
        
        # Create itterator object for the dataset
        self.data_iterator = iter(data_set)  # Reset the iterator if needed
        
        # Space definitions
        # NOTE: u must use an action wrapper to set self.action_space
        # NOTE: u must use an observation wrapper to set self.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None)
        
        # Reset map, path, init/goal pose, simulation and collision tree
        grid, proto_name, init_pose, goal_pose = self.get_data()
        print(f"Resetting the simulation with proto: {proto_name}")
        print(f"Init pose: {init_pose}")
        print(f"Goal pose: {goal_pose}")      
            
        # Resetting the simulation
        self.webots_env.reset()
        self.webots_env.reset_map(proto_name)
        # self.webots_env.reset_robot(init_pose)
        
        # self.collision_tree = et.compute_collision_detection_tree(grid, self.params['map_res'])
        # self.map_bounds_polygon = et.compute_map_bound_polygon(self.params)

        # # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        # self.cur_pos = self.webots_env.robot_position
        # self.cur_orient_matrix = self.webots_env.robot_orientation
        # self.footprint_glob = et.get_global_footprint_location(
        #     self.cur_pos, 
        #     self.cur_orient_matrix, 
        #     self.params['polygon_coords']
        # )
        # self.local_goal_pos = et.get_local_goal_pos(
        #     self.cur_pos, 
        #     self.cur_orient_matrix, 
        #     self.goal_pose
        # )
        # self.cur_vel, self.cmd_vel = np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
        # self.observation = self.get_obs()
        # # self.render(method='reset')

        # return self.observation, {} # = info

    # def step(self, action):
    #     # NOTE: u must use an action wrapper for turning the action vector into a cmd_vel
    #     self.cmd_vel = action

    #     # Updating previous values before updating
    #     self.prev_pos = self.cur_pos
    #     self.prev_observation = self.observation
           
    #     # Inacting the action (i.e. limit velocity to kinematically feasible values and forward simulate robot)
    #     self.cur_vel = et.apply_kinematic_constraints(self.params, self.cur_vel, self.cmd_vel)
    #     pos, orient = et.compute_new_pose(self.params, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        
    #     self.webots_env.step(pos, orient)

    #     # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
    #     self.prev_pos = self.cur_pos
    #     self.cur_pos = self.webots_env.robot_position
    #     self.cur_orient_matrix = self.webots_env.robot_orientation
    #     self.footprint_glob = et.get_global_footprint_location(
    #         self.cur_pos, 
    #         self.cur_orient_matrix, 
    #         self.params['polygon_coords']
    #     )
    #     self.local_goal_pos = et.get_local_goal_pos(
    #         self.cur_pos, 
    #         self.cur_orient_matrix, 
    #         self.goal_pose
    #     )

    #     self.observation = self.get_obs()
    #     self.done, self.done_cause = self.get_done()
    #     self.reward = self.reward(self.done, self.done_cause)
    #     # self.render(method='step')

    #     return self.observation, self.reward, self.done, False, {} # last 3: done, truncated, info
    
    def get_data(self):
        try:
            data_point = next(self.data_iterator)
        except StopIteration:
            print("No more data points available. Resetting iterator.")
            self.data_iterator = iter(self.data_set)
            self.get_data()
            
        return data_point['grid'], data_point['proto_name'], data_point['init_pose'], data_point['goal_pose']
    
    # def reward(self, done, done_cause):
    #     # NOTE: u must use a reward wrapper
    #     return 0.0 # = reward

    # def get_obs(self):
    #     # super().step(self.basic_timestep) # NOTE: moved to webots_env but might be needed here (weird bugfix)
    #     self.lidar_range_image = self.lidar_node.getRangeImage()

    #     return self.lidar_range_image
    
    # def get_done(self):
    #     done_cause = None

    #     # Arrived at the goal
    #     if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']):
    #         done_cause = 'at_goal'
    #         return True, done_cause
            
    #     # Driving outside map limit
    #     cur_pos_point = Point(self.cur_pos[0], self.cur_pos[1])
    #     if not (self.map_bounds_polygon.contains(cur_pos_point) or self.map_bounds_polygon.boundary.contains(cur_pos_point)):
    #         done_cause = 'outside_map'
    #         return True, done_cause
        
    #     # Collision with obstacles
    #     if len(self.collision_tree.query(footprint_glob, predicate='intersects')) > 0:
    #         done_cause = 'collision'
    #         return True, done_cause

    #     # When none of the done conditions are met
    #     return False, done_cause
    
    def close(self):
        self.webots_env.close()