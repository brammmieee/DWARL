import os
import random
from subprocess import Popen, PIPE
from controller import Supervisor, Keyboard
import numpy as np
import math as m
import matplotlib.pyplot as plt
import utils.base_tools as bt
import utils.admin_tools as at
import utils.wrappers.velocity_obstacle_tools as ovt
import gymnasium as gym
from matplotlib.patches import Polygon as plt_polygon
from shapely.geometry import Point, Polygon
from shapely.affinity import translate, rotate
from matplotlib import gridspec

class BaseEnv(Supervisor, gym.Env):
    metadata = {"render_modes": ["full", "position", "velocity", "trajectory"], "render_fps": 4} # TODO: use fps in render methods
    package_dir = os.path.abspath(os.pardir)
    
    def __init__(self, render_mode=None, wb_open=True, wb_mode='training', parameter_file='base_parameters.yaml', proto_config='default_proto_config.json'):
        # Directories
        self.resources_dir = os.path.join(BaseEnv.package_dir, 'resources')
        self.params_dir = os.path.join(BaseEnv.package_dir, 'parameters')
        self.paths_dir = os.path.join(self.resources_dir, 'paths')
        self.grids_dir = os.path.join(self.resources_dir, 'grids')
        self.worlds_dir = os.path.join(self.resources_dir, 'worlds')
        
        # Load parameters and set protos to configurations
        self.params = at.load_parameters([parameter_file, proto_config])
        bt.update_protos(proto_config)
        
        # Precomputations
        self.precomputed_lidar_values = bt.precompute_lidar_values(
            num_lidar_rays=self.params['proto_substitutions']['horizontalResolution']
        )

        # Training maps and map bounds
        self.train_map_nr_list = at.load_from_json('train_map_nr_list.json', os.path.join(self.params_dir, 'map_nrs'))
        self.map_bounds_polygon = bt.compute_map_bound_polygon(self.params)

        self.set_render_mode(render_mode)
        self.open_webots(wb_open, wb_mode)
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

        self.cur_vel = np.array([0.0, 0.0])
        self.cmd_vel = np.array([0.0, 0.0])
        
        self.method = 'reset' # used for rendering
        self.observation = self.get_obs()

        return self.observation, {} # = info

    def step(self, action):
        # Updating previous values before updating
        self.prev_pos = self.cur_pos
        self.prev_observation = self.observation
           
        # NOTE: u must use an action wrapper for turning the action vector inot a cmd_vel
        self.cmd_vel = action

        # Inacting the action
        self.cur_vel = self.cmd_vel
        pos, orient = bt.compute_new_pose(self.params, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        self.robot_translation_field.setSFVec3f([pos[0], pos[1], pos[2]])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, orient])
        
        super().step(self.basic_timestep) # WEBOTS - Step()

        # Updating prev_pos, cur_pos, cur_orient, footprint in global frame, and getting new local goal
        self.update_robot_state_and_local_goal(method='step')

        self.method = 'step'
        self.observation = self.get_obs()
        self.reward = self.get_reward()
        self.done = self.get_done()

        return self.observation, self.reward, self.done, False, {} # last 3: done, truncated, info

    def get_obs(self):
        # Getting lidar data and converting to pointcloud
        super().step(self.basic_timestep) #NOTE: only after this timestep will the lidar data of the previous step be available
        self.lidar_range_image = self.lidar_node.getRangeImage()
        self.render(method=self.method)

        return self.lidar_range_image

    def get_reward(self):
        # NOTE: u must use a reward wrapper to set a proper reward function
        reward = 0
        return reward

    def get_done(self):
        # Arrived at the goal
        if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']):
            return True
            
        # Driving outside map limit
        cur_pos_point = Point(self.cur_pos[0], self.cur_pos[1])
        if not (self.map_bounds_polygon.contains(cur_pos_point) or self.map_bounds_polygon.boundary.contains(cur_pos_point)):
            return True
        
        if self.check_collision():
            return True

        # When none of the done conditions are met
        return False
    
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
        result = self.collision_tree.query(self.footprint_glob)
        collision_detected = len(result) > 0

        if collision_detected:
            return True
        
        return collision_detected
    
    def close(self):
        self.close_webots()
                
    def open_webots(self, wb_open, wb_mode):
        if not wb_open:
            return
        world_file = os.path.join(self.worlds_dir, 'webots_world_file.wbt')
        if wb_mode == 'training':
            wb_process = Popen(['webots','--extern-urls', '--no-rendering', '--mode=fast', world_file], stdout=PIPE)
        elif wb_mode == 'testing':
            wb_process = Popen(['webots','--extern-urls', world_file], stdout=PIPE)
        else:
            print(f'init(): recieved invalid mode "{wb_mode}", should be either "training" or "testing"')
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
    
    def reset_map_path_and_poses(self):
        train_map_nr_idx = random.randint(0, len(self.train_map_nr_list)-1)
        self.map_nr = self.train_map_nr_list[train_map_nr_idx]
        self.grid = np.load(os.path.join(self.grids_dir, 'grid_' + str(self.map_nr) + '.npy'))
        path = np.load(os.path.join(self.paths_dir, 'path_' + str(self.map_nr) + '.npy'))
        self.path = np.multiply(path, self.params['map_res']) # apply scaling
        self.init_pose, self.goal_pose = bt.get_init_and_goal_poses(path=self.path, parameters=self.params) # pose -> [x,y,psi]

    def set_render_mode(self, render_mode):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_count = 0

    def render(self, method=None):
        if self.render_mode == None:
            return
        
        # Create initial plot or remove data
        if self.render_count == 0:
            self.render_init_plot(self.render_mode)
        else:
            self.render_remove_data(self.render_mode, method)
            
        # Add new data to the plot
        self.render_add_data(self.render_mode, method)
        
        # Plot graphs set flags and counter
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.render_count += 1 # counter for reduced rendering (e.g. every 2nd step)
        
    def render_init_plot(self, render_mode):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        
        if render_mode in ['position','full']:
            # ax1 - Lidar data and footprint (axis fixed to base_link)
            polygon = Polygon(self.params['polygon_coords'])
            patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.75, closed=True, facecolor='grey')
            self.ax1.add_patch(patch)

            self.ax1.set_xlim([-1.5,1.5])
            self.ax1.set_ylim([-1.5,1.5])
            self.ax1.set_xlabel('x [m]')
            self.ax1.set_ylabel('y [m]')
            self.ax1.set_aspect('equal')
            self.ax1.grid()
            
        if render_mode in ['trajectory', 'full']:
            # ax2 - Path, init pose and goal pose data (axis fixed to map)
            self.ax2.set_aspect('equal', adjustable='box')
            self.ax2.set_xlabel('x [m]')
            self.ax2.set_ylabel('y [m]')
        
    def render_remove_data(self, render_mode, method):
        if render_mode in ['position', 'full']:
            try:
                # ax1 - Clear lidar data
                self.lidar_plot.remove()
                self.local_goal_plot.remove()
            except AttributeError:
                pass

        if render_mode in ['trajectory', 'full']:
            try:
                # ax2 - Clear path and poses
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

    def render_add_data(self, render_mode, method):
        if render_mode in ['position', 'full']:
            # ax1 - Lidar data
            self.lidar_points = bt.lidar_to_point_cloud(self.params, self.precomputed_lidar_values, self.lidar_range_image) #NOTE: double computation in case of e.g vo observation wrapper
            self.lidar_plot = self.ax1.scatter(self.lidar_points[:,0], self.lidar_points[:,1], alpha=1.0, c='black')
            self.local_goal_plot = self.ax1.scatter(self.local_goal_pos[0], self.local_goal_pos[1], alpha=1.0, c='purple')
            
        if render_mode in ['trajectory', 'full']:
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