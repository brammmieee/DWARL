import os
import random
import numpy as np
import math as m
import gymnasium as gym
from subprocess import Popen, PIPE
from controller import Supervisor, Keyboard
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon
from shapely.affinity import translate, rotate
from scipy.interpolate import interp1d

import utils.env_tools as et
import utils.admin_tools as at

class BaseEnv(Supervisor, gym.Env):
    package_dir = os.path.abspath(os.pardir)
    
    def __init__(self, cfg):        
        self.cfg = cfg
        # self.render_canvas_drawn = False

        # Update the proto files according to the configuration
        et.update_protos(self.cfg.proto_reconfiguration)
        
        # Precomputations
        self.precomputed_lidar_values = et.precompute_lidar_values(
            num_lidar_rays=self.cfg.env.proto_reconfiguration.proto_substitutions.horizontalResolution
        )

        # Training maps and map bounds
        self.train_map_nr_list = at.load_from_json('train_map_nr_list.json', os.path.join(self.params_dir, 'map_nrs'))
        self.map_bounds_polygon = et.compute_map_bound_polygon(self.params)

        et.open_webots(cfg.webots)
        et.init_webots()

        # Space definitions
        # NOTE: u must use an action wrapper to set self.action_space
        # NOTE: u must use an observation wrapper to set self.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None)
        
        # Reset map, path, init/goal pose and simulation
        self.reset_map_path_and_poses()
        self.reset_webots()

        # Compute collision tree for efficient collision detection (depends on new grid)
        self.collision_tree = et.precompute_collision_detection(self.grid, self.params['map_res'])

        # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        self.update_robot_state_and_local_goal(method='reset')

        # Resetting the path progress
        self.reset_path_dist_progress_and_heading()

        self.cur_vel = np.array([0.0, 0.0])
        self.cmd_vel = np.array([0.0, 0.0])
        
        self.observation = self.get_obs()
        if self.cfg.render:
            self.render(method='reset')

        return self.observation, {} # = info

    def step(self, action):
        # Updating previous values before updating
        self.prev_pos = self.cur_pos
        self.prev_observation = self.observation
           
        if self.teleop == True:
            action = et.get_teleop_action(self.keyboard)

        # NOTE: u must use an action wrapper for turning the action vector inot a cmd_vel
        self.cmd_vel = action

        # Inacting the action (i.e. limit velocity to kinematically feasible values and forward simulate robot)
        self.cur_vel = et.apply_kinematic_constraints(self.params, self.cur_vel, self.cmd_vel)
        pos, orient = et.compute_new_pose(self.params, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        self.robot_translation_field.setSFVec3f([pos[0], pos[1], pos[2]])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, orient])
        
        super().step(self.basic_timestep) # WEBOTS - Step()

        # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        self.update_robot_state_and_local_goal(method='step')
        self.update_path_dist_progress_and_heading()

        self.observation = self.get_obs()
        self.done, done_cause = self.get_done()
        self.reward = self.reward(self.done, done_cause)

        if self.cfg.render:
            self.render(method='step')

        return self.observation, self.reward, self.done, False, {} # last 3: done, truncated, info
    
    def reward(self, done, done_cause):
        # NOTE: u must use a reward wrapper
        return 0.0 # = reward

    def get_obs(self):
        # Getting lidar data and converting to pointcloud
        super().step(self.basic_timestep) #NOTE: only after this timestep will the lidar data of the previous step be available
        self.lidar_range_image = self.lidar_node.getRangeImage()

        return self.lidar_range_image
    
    def get_done(self):
        done_cause = None

        # Arrived at the goal
        if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']):
            done_cause = 'at_goal'
            return True, done_cause
            
        # Driving outside map limit
        cur_pos_point = Point(self.cur_pos[0], self.cur_pos[1])
        if not (self.map_bounds_polygon.contains(cur_pos_point) or self.map_bounds_polygon.boundary.contains(cur_pos_point)):
            done_cause = 'outside_map'
            return True, done_cause
        
        if self.check_collision():
            done_cause = 'collision'
            return True, done_cause

        # When none of the done conditions are met
        return False, done_cause

    def reset_path_dist_progress_and_heading(self):
        self.path_dist = 0
        self.prev_path_dist = 0

        self.path_progress = 0
        self.prev_path_progress = 0

        self.path_heading = 0
        self.prev_path_heading = 0

        self.init_progress = self.calculate_progress(self.init_pose)
        self.goal_progress = self.calculate_progress(self.goal_pose)

    def reset_map_path_and_poses(self):
        train_map_nr_idx = random.randint(0, len(self.train_map_nr_list)-1)
        self.map_nr = self.train_map_nr_list[train_map_nr_idx]
        self.grid = np.load(os.path.join(self.grids_dir, 'grid_' + str(self.map_nr) + '.npy'))
        path = np.load(os.path.join(self.paths_dir, 'path_' + str(self.map_nr) + '.npy'))
        self.path = np.multiply(path, self.params['map_res']) # apply scaling

        self.init_pose, self.goal_pose, self.direction = et.get_init_and_goal_poses(path=self.path, parameters=self.params) # pose -> [x,y,psi]

    def update_robot_state_and_local_goal(self, method):
        # Update previous position, current position, current orientation, and global footprint location
        if method == 'step':
            self.prev_pos = np.array(self.cur_pos)
        self.cur_pos = np.array(self.robot_node.getPosition())
        self.cur_orient_matrix = np.array(self.robot_node.getOrientation())
        self.footprint_glob = self.get_global_footprint_location(
            self.cur_pos, self.cur_orient_matrix, self.params['polygon_coords']
        )
        
        # Calculate local goal position
        self.local_goal_pos = et.get_local_goal_pos(self.cur_pos, self.cur_orient_matrix, self.goal_pose)

    def get_global_footprint_location(self, cur_pos, cur_orient_matrix, polygon_coords):
        # Get position and orientation
        position = cur_pos[:2]
        orientation_matrix = cur_orient_matrix

        # Construct translated and rotated polygon
        footprint = Polygon(polygon_coords)
        translated_footprint = translate(footprint, position[0], position[1])
        angle_rad = m.atan2(orientation_matrix[3], orientation_matrix[0])  # atan2(sin, cos)
        rotated_footprint = rotate(
            translated_footprint, 
            angle_rad-0.5*np.pi,
            origin=Point(position[0], position[1]), 
            use_radians=True
        )
        
        return rotated_footprint
    
    def check_collision(self):
        # Use the STRTree to find any intersecting boxes
        result = self.collision_tree.query(self.footprint_glob, predicate='intersects')
        collision_detected = len(result) > 0

        if collision_detected:
            return True
        
        return collision_detected
    
    def close(self):
        et.close_webots()

    # def render(self, method=None):
    #     # Initialize the plot or remove the previous data 
    #     if self.render_canvas_drawn == False:
    #         self.render_init_plot()
    #     else:
    #         self.render_remove_data(method)
            
    #     # Add new data to the plot
    #     self.render_add_data(method)
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()
        
    # def render_init_plot(self):
    #     self.render_canvas_drawn = True

    #     plt.ion()
    #     self.fig, ((self.ax1, self.ax3), (self.ax2, self.ax4)) = plt.subplots(2, 2)
        
    #     # ax1 - Lidar data and footprint (axis fixed to base_link)
    #     polygon = Polygon(self.params['polygon_coords'])
    #     patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.75, closed=True, facecolor='grey')
    #     self.ax1.add_patch(patch)
    #     self.ax1.set_xlim([-3.0, 3.0])
    #     self.ax1.set_ylim([-3.0, 3.0])
    #     self.ax1.set_xlabel('x [m]')
    #     self.ax1.set_ylabel('y [m]')
    #     self.ax1.set_aspect('equal')
    #     self.ax1.grid()
            
    #     # ax2 - Path, init pose and goal pose data (axis fixed to map)
    #     self.ax2.set_aspect('equal', adjustable='box')
    #     self.ax2.set_xlabel('x [m]')
    #     self.ax2.set_ylabel('y [m]')

    #     # Assuming self.reward_style_map is defined as before
    #     legend_handles = [
    #         Line2D([0], [0], color=style['color'], linestyle=style['linestyle'], label=label)
    #         for label, style in self.reward_style_map.items()
    #     ]

    #     # Adjust the legend in your plotting method
    #     self.fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=self.fig.transFigure)
    
    # def render_add_data(self, method):
    #     # ax1 - Lidar data
    #     self.lidar_points = et.lidar_to_point_cloud(self.params, self.precomputed_lidar_values, self.lidar_range_image) #NOTE: double computation in case of e.g vo observation wrapper
    #     self.lidar_plot = self.ax1.scatter(self.lidar_points[:,0], self.lidar_points[:,1], alpha=1.0, c='black')
    #     self.local_goal_plot = self.ax1.scatter(self.local_goal_pos[0], self.local_goal_pos[1], alpha=1.0, c='purple')
            
    #     # ax2 - Path and poses
    #     self.cur_pos_plot = self.ax2.scatter(self.cur_pos[0], self.cur_pos[1], c='blue', alpha=0.33)
    #     x, y = self.footprint_glob.exterior.xy
    #     self.footprint_plot = self.ax2.fill(x, y, color='blue', alpha=0.5)
    #     if method == 'reset':
    #         self.grid_plots = []  # Initialize a list to store all rectangle patches
    #         indices = np.argwhere(self.grid == 1)
    #         for index in indices:
    #             x, y = index
    #             x_scaled, y_scaled = x * self.params['map_res'], y * self.params['map_res']
    #             rect = plt.Rectangle((x_scaled, y_scaled), self.params['map_res'], self.params['map_res'], color='black')
    #             self.grid_plots.append(self.ax2.add_patch(rect))  # Add each patch to the list

    #         self.path_plot = self.ax2.scatter(self.path[:,0], self.path[:,1], c='grey', alpha=0.5)
    #         self.init_pose_plot = self.ax2.scatter(self.init_pose[0], self.init_pose[1], c='green')
    #         self.goal_pose_plot = self.ax2.scatter(self.goal_pose[0], self.goal_pose[1], c='red')

    # def render_remove_data(self, method):
    #     # ax1 - Clear lidar data
    #     try:
    #         self.lidar_plot.remove()
    #         self.local_goal_plot.remove()
    #     except AttributeError:
    #         pass

    #     # ax2 - Clear path and poses
    #     try:
    #         self.cur_pos_plot.remove()
    #         for patch in self.footprint_plot:
    #             patch.remove()
    #         if method == 'reset':
    #             for patch in self.grid_plots:
    #                 patch.remove()
    #             self.grid_plots.clear()
    #             self.path_plot.remove()
    #             self.init_pose_plot.remove()
    #             self.goal_pose_plot.remove()
    #     except AttributeError:
    #         pass

