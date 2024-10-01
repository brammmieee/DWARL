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

import utils.base_tools as bt
import utils.admin_tools as at

class BaseEnv(Supervisor, gym.Env):
    package_dir = os.path.abspath(os.pardir)
    
    def __init__(self, render_mode=None, wb_open=True, 
                 wb_mode='training', parameter_file='base_parameters.yaml', 
                 proto_config='default_proto_config.json',
                 reward_config='parameterized_reward.yaml', 
                 wb_headless=False, 
                 teleop=False,
                 plot_wrapped_obs=False
                 ):
        
        # Directories
        self.resources_dir = os.path.join(BaseEnv.package_dir, 'resources')
        self.params_dir = os.path.join(BaseEnv.package_dir, 'parameters')
        self.paths_dir = os.path.join(self.resources_dir, 'paths')
        self.grids_dir = os.path.join(self.resources_dir, 'grids')
        self.worlds_dir = os.path.join(self.resources_dir, 'worlds')
        
        # Load parameters and set protos to configurations
        self.params = at.load_parameters([parameter_file, proto_config, reward_config])
        bt.update_protos(proto_config)
        
        # Precomputations
        self.precomputed_lidar_values = bt.precompute_lidar_values(
            num_lidar_rays=self.params['proto_substitutions']['horizontalResolution']
        )

        # Wrapper flags
        self.plot_wrapped_obs = plot_wrapped_obs

        # Teleoperation
        self.teleop = teleop

        # Reward buffer and plotting
        reward_plotting_config = at.load_parameters(['reward_plotting_config.json'])
        self.reward_plot_map = {key: value["plot_nr"] for key, value in reward_plotting_config.items()}

        self.reward_buffers = {component: [] for component in reward_plotting_config.keys()}
        self.reward_style_map = {key: value["plot_style"] for key, value in reward_plotting_config.items() if self.reward_plot_map.get(key, 0) != 0}

        # Training maps and map bounds
        self.train_map_nr_list = at.load_from_json('train_map_nr_list.json', os.path.join(self.params_dir, 'map_nrs'))
        self.map_bounds_polygon = bt.compute_map_bound_polygon(self.params)

        self.set_render_mode(render_mode)
        self.open_webots(wb_open, wb_mode, wb_headless)
        self.init_webots()

        # Space definitions
        # NOTE: u must use an action wrapper to set self.action_space
        # NOTE: u must use an observation wrapper to set self.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None)
        
        # Reset map, path, init/goal pose and simulation
        self.reset_map_path_and_poses()
        self.reset_webots()

        # Compute collision tree for efficient collision detection (depends on new grid)
        self.collision_tree = bt.precompute_collision_detection(self.grid, self.params['map_res'])

        # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        self.update_robot_state_and_local_goal(method='reset')

        # Resetting the path progress
        self.reset_path_dist_progress_and_heading()

        self.cur_vel = np.array([0.0, 0.0])
        self.cmd_vel = np.array([0.0, 0.0])
        
        self.observation = self.get_obs()
        self.render(method='reset')

        return self.observation, {} # = info

    def step(self, action):
        # Updating previous values before updating
        self.prev_pos = self.cur_pos
        self.prev_observation = self.observation
           
        if self.teleop == True:
            action = bt.get_teleop_action(self.keyboard)

        # NOTE: u must use an action wrapper for turning the action vector inot a cmd_vel
        self.cmd_vel = action

        # Inacting the action (i.e. limit velocity to kinematically feasible values and forward simulate robot)
        self.cur_vel = bt.apply_kinematic_constraints(self.params, self.cur_vel, self.cmd_vel)
        pos, orient = bt.compute_new_pose(self.params, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        self.robot_translation_field.setSFVec3f([pos[0], pos[1], pos[2]])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, orient])
        
        super().step(self.basic_timestep) # WEBOTS - Step()

        # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        self.update_robot_state_and_local_goal(method='step')
        self.update_path_dist_progress_and_heading()

        self.observation = self.get_obs()
        self.done, done_cause = self.get_done()
        self.reward = self.get_reward(self.done, done_cause)

        self.render(method='step')

        return self.observation, self.reward, self.done, False, {} # last 3: done, truncated, info

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

    def get_reward(self, done, done_cause):    
        # Initialize reward components dictionary with empty values
        reward_components = {
            'r_at_goal': 0,
            'r_outside_map': 0,
            'r_collision': 0,
            'r_not_arrived': 0,
            'r_path_d_dist': 0,
            'r_path_d_progress': 0,
            'r_linear_d_progress': 0,
            'r_path_d_heading': 0,
            'r_path_dist': 0,
            'r_path_heading': 0,
            'r_path_d_progress_scaled_dist': 0,
        }

        if done:
            # Assign reward based on the done cause
            if done_cause == 'at_goal':
                reward_components['r_at_goal'] = self.params['r_at_goal']
            elif done_cause == 'outside_map':
                reward_components['r_outside_map'] = self.params['r_outside_map']
            elif done_cause == 'collision':
                reward_components['r_collision'] = self.params['r_collision']
            else:
                raise ValueError(f'get_reward(): done_cause "{done_cause}" not recognized')
        else:
            # Calculate ongoing rewards
            reward_components['r_not_arrived'] = self.params['r_not_arrived']

            reward_components['r_path_d_dist'] = self.params['c_path_d_dist']*self.get_normalized_path_d_dist()
            normalized_path_progress = self.get_normalized_path_d_progress()
            reward_components['r_path_d_progress'] = self.params['c_path_d_progress']*normalized_path_progress
            reward_components['r_linear_d_progress'] = self.params['c_linear_d_progress']*self.get_normalized_linear_d_progress()
            reward_components['r_path_d_heading'] = self.params['c_path_d_heading']*self.get_normalized_path_d_heading()
            
            normalized_path_dist_reward = self.get_normalized_path_dist_reward()
            reward_components['r_path_dist'] = self.params['c_path_dist']*normalized_path_dist_reward
            reward_components['r_path_heading'] = self.params['c_path_heading']*self.get_normalized_path_heading_reward()
            reward_components['r_path_d_progress_scaled_dist'] = self.params['c_path_d_progress_scaled_dist']*self.get_normalized_d_pogress_scaled_path_dist_reward(normalized_path_progress, normalized_path_dist_reward)
            
        # Calculate total reward as the sum of all components
        total_reward = sum(reward_components.values())

        # Append rewards to reward buffers for plotting
        if self.render_mode is not None:
            self.update_reward_buffers(reward_components, total_reward)

        return total_reward
        
    def get_normalized_path_d_dist(self):
        if self.params['c_path_d_dist'] == 0:
            return 0    
        path_dist_diff = -(self.path_dist - self.prev_path_dist)
        max_path_dist = self.params['v_max']*self.params['sample_time']
        normalized_path_dist = np.clip(path_dist_diff / max_path_dist, -1, 1)

        return normalized_path_dist
    
    def get_normalized_path_d_progress(self):

        if self.path_progress is not None and self.prev_path_progress is not None:
            if self.params['c_path_d_progress'] == 0 and self.params['c_path_d_progress_scaled_dist'] == 0:
                return 0
            path_progress_diff = self.path_progress - self.prev_path_progress
            max_path_progress = self.params['v_max']*self.params['sample_time']

            if not self.params['enable_d_progress_mapping']:
                normalized_path_progress_diff = path_progress_diff / max_path_progress
            else:
                normalized_path_progress_diff = self.get_mapped_normalized_progress_diff(path_progress_diff)

            normalized_path_progress = np.clip(normalized_path_progress_diff, -1, 1)
        else: # Robot is not between init and goal
            normalized_path_progress = 0

        return normalized_path_progress
    
    def get_normalized_linear_d_progress(self):
        if self.params['c_linear_d_progress'] == 0:
            return 0
        prev_dist_to_goal = np.linalg.norm(self.goal_pose[:2] - self.prev_pos[:2])
        current_dist_to_goal = np.linalg.norm(self.goal_pose[:2] - self.cur_pos[:2])
        progress_diff = prev_dist_to_goal - current_dist_to_goal
        max_progress_diff = self.params['v_max']*self.params['sample_time']
        normalized_progress_diff = np.clip(progress_diff / max_progress_diff, -1, 1)
        
        return normalized_progress_diff

    def get_mapped_normalized_progress_diff(self, path_progress_diff):
        # print(f"x: {path_progress_diff}")
        if path_progress_diff < self.params['d_progress_low_pass_x']:
            progress_diff = -1
        elif path_progress_diff > self.params['d_progress_high_pass_x']:
            progress_diff = 1
        else:
            x = path_progress_diff
            progress_diff = -674663.913519*x**3 + 5658.877973*x**2 + 213.837054*x**1 - 1.029126
        # print(f"f(x): {progress_diff}")
        return progress_diff
            
    def get_normalized_path_d_heading(self):

        if self.path_heading is not None and self.prev_path_heading is not None:
            if self.params['c_path_d_heading'] == 0:
                return 0
            path_heading_diff = -(self.path_heading - self.prev_path_heading)
            max_path_heading = self.params['omega_max']*self.params['sample_time']
            normalized_path_heading = np.clip(path_heading_diff / max_path_heading, -1, 1)
        else: # Robot is not between init and goal
            normalized_path_heading = 0

        return normalized_path_heading
    
    def get_normalized_path_dist_reward(self):
        if self.params['c_path_dist'] == 0 and self.params['c_path_d_progress_scaled_dist'] == 0:
            return 0    
        max_distance = self.params['max_path_dist']
        normalized_path_dist = self.path_dist / max_distance
        normalized_path_dist_reward = np.clip(1 - normalized_path_dist, -1, 1)
        
        return normalized_path_dist_reward

    def get_normalized_path_heading_reward(self):
        if self.path_heading is not None and self.prev_path_heading is not None:
            if self.params['c_path_heading'] == 0:
                return 0
            max_path_heading = self.params['max_path_heading']
            normalized_path_heading = self.path_heading / max_path_heading
            normalized_path_heading_reward = np.clip(1 - normalized_path_heading, -1, 1)
        else: # Robot is not between init and goal
            normalized_path_heading_reward = 0

        return normalized_path_heading_reward
    
    def get_normalized_d_pogress_scaled_path_dist_reward(self, normalized_path_progress, normalized_path_dist_reward):
        if self.params['c_path_d_progress_scaled_dist'] == 0:
            return 0
        
        if np.clip(normalized_path_progress, 0, 1) >= 0.0:
            return np.clip(normalized_path_progress, 0, 1)*normalized_path_dist_reward
        else:
            return normalized_path_dist_reward
        
    def update_reward_buffers(self, reward_components, total_reward):
        for key, value in reward_components.items():
            self.reward_buffers[key].append(value)
            if len(self.reward_buffers[key]) > self.params['reward_buffer_size']:
                self.reward_buffers[key].pop(0)
                
        self.reward_buffers['total_reward'].append(total_reward)
        if len(self.reward_buffers['total_reward']) > self.params['reward_buffer_size']:
            self.reward_buffers['total_reward'].pop(0)

    def update_path_dist_progress_and_heading(self):
        path_dist = np.inf  # Distance to the path
        path_heading = 0
        cumulative_path_length = 0
        r = self.cur_pos[:2]
        psi = (np.arctan2(self.cur_orient_matrix[3], self.cur_orient_matrix[0])) % (2*np.pi)

        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i + 1])
            v = p2 - p1
            w = r - p1
            t = np.dot(w, v) / np.dot(v, v)

            # Check if robot is still between init and goal
            if i == 0 and t < 0:  # Robot is before starting point (init) of the path
                path_progress = None
                path_heading = None
                path_dist = np.linalg.norm(r - self.path[0])
                break
            elif i == len(self.path) - 2 and t > 1:  # Robot has passed the end point (goal) of the path
                path_progress = None
                path_heading = None
                path_dist = np.linalg.norm(r - self.path[-1])
                break
            else:  # Robot is between init and goal

                t = max(0, min(1, t))
                closest_point_on_segment = p1 + t * v
                segment_dist = np.linalg.norm(r - closest_point_on_segment)

                # Caculation of path_dist, path_progress, and path_heading
                if segment_dist < path_dist:
                    # Path dist calculation
                    path_dist = segment_dist

                    # Path progress calculation
                    path_progress = cumulative_path_length + t * np.linalg.norm(v)  # Progress along the path

                    # Path heading calculation
                    path_angle = np.arctan2(v[1], v[0])
                    path_angle = path_angle % (2*np.pi)
                    path_heading = np.abs(psi - path_angle)
                    self.t = t

                cumulative_path_length += np.linalg.norm(v)

        self.prev_path_dist = self.path_dist
        self.path_dist = path_dist if path_dist < np.inf else 0
        self.prev_path_progress = self.path_progress
        self.path_progress = path_progress
        self.prev_path_heading = self.path_heading
        self.path_heading = path_heading

    def calculate_progress(self, pose):
        total_length = 0
        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i + 1])
            if np.array_equal(p1, pose[:2]):
                progress = total_length
                break
            if np.array_equal(p2, pose[:2]):
                progress = total_length + np.linalg.norm(p2 - p1)
                break
            total_length += np.linalg.norm(p2 - p1)

        return progress

    def reset_path_dist_progress_and_heading(self):
        self.path_dist = 0
        self.prev_path_dist = 0

        self.path_progress = 0
        self.prev_path_progress = 0

        self.path_heading = 0
        self.prev_path_heading = 0

    def reset_map_path_and_poses(self):
        train_map_nr_idx = random.randint(0, len(self.train_map_nr_list)-1)
        self.map_nr = self.train_map_nr_list[train_map_nr_idx]
        self.grid = np.load(os.path.join(self.grids_dir, 'grid_' + str(self.map_nr) + '.npy'))
        path = np.load(os.path.join(self.paths_dir, 'path_' + str(self.map_nr) + '.npy'))
        path = path[1:]  # For some reason, the first two points of the path are duplicates
        path = np.multiply(path, self.params['map_res']) # apply scaling

        self.init_pose, self.goal_pose, self.path = bt.get_init_and_goal_poses(path=path, parameters=self.params) # pose -> [x,y,psi]

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
        self.local_goal_pos = bt.get_local_goal_pos(self.cur_pos, self.cur_orient_matrix, self.goal_pose)

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
        self.close_webots()
                
    def open_webots(self, wb_open, wb_mode, wb_headless):
        if not wb_open:
            return
        
        # Create Webots command with specified mode and world file
        world_file = os.path.join(self.worlds_dir, 'webots_world_file.wbt')
        if wb_mode == 'training':
            cmd = ['webots','--extern-urls', '--no-rendering', '--mode=fast', world_file]
        elif wb_mode == 'testing':
            cmd = ['webots','--extern-urls', world_file]
        else:
            print(f'init(): recieved invalid mode "{wb_mode}", should be either "training" or "testing"')
            return
        
        # Use X virtual framebuffer (Xvfb) to Webots in headless mode
        if wb_headless:
            cmd = ['xvfb-run', '--auto-servernum'] + cmd

        # Open Webots
        wb_process = Popen(cmd, stdout=PIPE)

        # Set the environment variable for the controller to connect to the supervisor
        output = wb_process.stdout.readline().decode("utf-8")
        ipc_prefix = 'ipc://'
        start_index = output.find(ipc_prefix)
        port_nr = output[start_index + len(ipc_prefix):].split('/')[0]
        os.environ["WEBOTS_CONTROLLER_URL"] = ipc_prefix + str(port_nr)
    
    def close_webots(self):
        self.simulationQuit(0)
        
    def init_webots(self):
        # Static node references
        super().__init__() # Env class instance (self) inherits Supervisor's methods + Robot's methods
        self.basic_timestep = int(self.getBasicTimeStep())
        self.timestep = 2*self.basic_timestep #NOTE: basic timestep set 0.5*timestep for lidar update
        self.robot_node = self.getFromDef('ROBOT')
        self.root_node = self.getRoot() # root node (the nodes seen in the scene tree editor window are children of the root node)
        self.robot_translation_field = self.robot_node.getField('translation')
        self.robot_rotation_field = self.robot_node.getField('rotation')
        self.root_children_field = self.root_node.getField('children') # used for inserting map node

        # Lidar sensor and keyboard
        self.lidar_node = self.getDevice('lidar')
        self.lidar_node.enable(int(self.getBasicTimeStep()))
        self.keyboard = Keyboard()

    def reset_webots(self):
        self.simulationReset()
        super().step(self.basic_timestep) # super prevents confusion with self.step() defined below
        
        # Loading and translating map into position
        self.root_children_field.importMFNodeFromString(position=-1, nodeString='DEF MAP ' + 'yaml_' + str(self.map_nr) + '{}')
        map_node = self.getFromDef('MAP')
        map_node_translation_field = map_node.getField('translation')
        map_node_translation_field.setSFVec3f([self.params['map_res']*self.params['map_width'], -(self.params['map_res']*self.params['map_width'])-3*self.params['map_res'], 0.0])
        super().step(self.basic_timestep)

        # Positioning the robot at init_pos
        self.robot_translation_field.setSFVec3f([self.init_pose[0], self.init_pose[1], self.params['z_pos']]) #TODO: add z_pos to init_pose[2]
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, self.init_pose[3]])
        super().step(2*self.basic_timestep) #NOTE: 2 timesteps needed in order to succesfully set the init position
    
    def set_render_mode(self, render_mode):
        self.render_mode = render_mode
        self.render_count = 0

    def render(self, method=None):
        if self.render_mode == None:
            return
        
        # Create initial plot or remove data
        if self.render_count == 0:
            self.render_init_plot()
        else:
            self.render_remove_data(method)
            
        # Add new data to the plot
        self.render_add_data(method)
        
        # Plot graphs set flags and counter
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.render_count += 1 # counter for reduced rendering (e.g. every 2nd step)
        
    def render_init_plot(self):
        plt.ion()
        self.fig, ((self.ax1, self.ax3), (self.ax2, self.ax4)) = plt.subplots(2, 2)
        
        # ax1 - Lidar data and footprint (axis fixed to base_link)
        polygon = Polygon(self.params['polygon_coords'])
        patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.75, closed=True, facecolor='grey')
        self.ax1.add_patch(patch)
        self.ax1.set_xlim([-3.0, 3.0])
        self.ax1.set_ylim([-3.0, 3.0])
        self.ax1.set_xlabel('x [m]')
        self.ax1.set_ylabel('y [m]')
        self.ax1.set_aspect('equal')
        self.ax1.grid()
            
        # ax2 - Path, init pose and goal pose data (axis fixed to map)
        self.ax2.set_aspect('equal', adjustable='box')
        self.ax2.set_xlabel('x [m]')
        self.ax2.set_ylabel('y [m]')

        # ax3 - Reward plot
        self.ax3.set_xlabel('Step')
        self.ax3.set_ylabel('Reward')
        self.ax3.set_ylim(-15, 15)
        self.ax3.grid()
        self.reward_plots_1 = {}

        # ax4 - dReward/dStep plot
        self.ax4.set_xlabel('Step')
        self.ax4.set_ylabel('Reward')
        self.ax4.set_ylim(-15, 15)
        self.ax4.grid()
        self.reward_plots_2 = {}

        # Assuming self.reward_style_map is defined as before
        legend_handles = [
            Line2D([0], [0], color=style['color'], linestyle=style['linestyle'], label=label)
            for label, style in self.reward_style_map.items()
        ]

        # Adjust the legend in your plotting method
        self.fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=self.fig.transFigure)
    
    def render_add_data(self, method):
        # ax1 - Lidar data
        self.lidar_points = bt.lidar_to_point_cloud(self.params, self.precomputed_lidar_values, self.lidar_range_image) #NOTE: double computation in case of e.g vo observation wrapper
        self.lidar_plot = self.ax1.scatter(self.lidar_points[:,0], self.lidar_points[:,1], alpha=1.0, c='black')
        self.local_goal_plot = self.ax1.scatter(self.local_goal_pos[0], self.local_goal_pos[1], alpha=1.0, c='purple')
            
        # ax2 - Path and poses
        self.cur_pos_plot = self.ax2.scatter(self.cur_pos[0], self.cur_pos[1], c='blue', alpha=0.33)
        x, y = self.footprint_glob.exterior.xy
        self.footprint_plot = self.ax2.fill(x, y, color='blue', alpha=0.5)
        if method == 'reset':
            self.grid_plots = []  # Initialize a list to store all rectangle patches
            indices = np.argwhere(self.grid == 1)
            for index in indices:
                x, y = index
                x_scaled, y_scaled = x * self.params['map_res'], y * self.params['map_res']
                rect = plt.Rectangle((x_scaled, y_scaled), self.params['map_res'], self.params['map_res'], color='black')
                self.grid_plots.append(self.ax2.add_patch(rect))  # Add each patch to the list

            self.path_plot = self.ax2.scatter(self.path[:,0], self.path[:,1], c='grey', alpha=0.5)
            self.init_pose_plot = self.ax2.scatter(self.init_pose[0], self.init_pose[1], c='green')
            self.goal_pose_plot = self.ax2.scatter(self.goal_pose[0], self.goal_pose[1], c='red')

        # ax3 - Reward plot
        for component, buffer in self.reward_buffers.items():
            if self.reward_plot_map[component] != 1:
                continue
            style = self.reward_style_map.get(component, {'color': 'gray', 'linestyle': ':'})  # Default style
            self.reward_plots_1[component] = self.ax3.plot(range(len(buffer)), buffer, label=component, **style)

        # ax4 - dReward/dStep plot
        for component, buffer in self.reward_buffers.items():
            if self.reward_plot_map[component] != 2:
                continue
            style = self.reward_style_map.get(component, {'color': 'gray', 'linestyle': ':'})  # Default style
            self.reward_plots_2[component] = self.ax4.plot(range(len(buffer)), buffer, label=component, **style)

    def render_remove_data(self, method):
        # ax1 - Clear lidar data
        try:
            self.lidar_plot.remove()
            self.local_goal_plot.remove()
        except AttributeError:
            pass

        # ax2 - Clear path and poses
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

        # ax3 - Clear reward plot
        try:
            for reward_plot in self.reward_plots_1.values():
                for plot in reward_plot:
                    plot.remove()
        except AttributeError:
            pass

        # ax4 - Clear dReward/dStep plot
        try:
            for reward_plot in self.reward_plots_2.values():
                for plot in reward_plot:
                    plot.remove()
        except AttributeError:
            pass
