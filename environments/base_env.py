import sys
import os
import yaml
import random
from subprocess import Popen, PIPE
from controller import Supervisor, Keyboard
import numpy as np
import matplotlib.pyplot as plt
import utils.general_tools as gt
import utils.admin_tools as at
import utils.wrappers.obstacle_velocity_observation_tools as ovt
import gymnasium as gym
from matplotlib.patches import Polygon as plt_polygon
from shapely.geometry import Point, Polygon
from matplotlib import gridspec

class BaseEnv(Supervisor, gym.Env):
    metadata = {"render_modes": ["full", "position", "velocity", "trajectory"], "render_fps": 4} # TODO: use fps in render methods
    package_dir = os.path.abspath(os.pardir)
    
    def __init__(self, render_mode=None, wb_open=True, wb_mode='training', reward_monitoring=False):
        # Directories
        self.resources_dir = os.path.join(BaseEnv.package_dir, 'resources')
        self.params_dir = os.path.join(BaseEnv.package_dir, 'parameters')
        self.paths_dir = os.path.join(self.resources_dir, 'paths')
        self.grids_dir = os.path.join(self.resources_dir, 'grids')
        self.worlds_dir = os.path.join(self.resources_dir, 'worlds')
        
        # Parameters
        self.params = at.load_parameters("general_parameters.yaml")
        
        # Precomputations # TODO: Check if can be removed
        self.precomputed_lidar_values = gt.precompute_lidar_values(self.params)

        # Training maps and map bounds
        self.train_map_nr_list = at.load_from_json('train_map_nr_list.json', os.path.join(self.params_dir, 'map_nrs'))
        self.map_bounds_polygon = gt.compute_map_bound_polygon(self.params)

        self.set_render_mode(render_mode)
        self.open_webots(wb_open, wb_mode)
        self.init_webots()

        # Space definitions
        # NOTE: u must use an action wrapper to set self.action_space
        # NOTE: u must use an observation wrapper to set self.observation_space

    def reset(self, seed=None):
        # TODO: super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None)
        
        # Setting map, path and init and goal pose
        train_map_nr_idx = random.randint(0, len(self.train_map_nr_list)-1)
        self.map_nr = self.train_map_nr_list[train_map_nr_idx]
        path = np.load(os.path.join(self.paths_dir, 'path_' + str(self.map_nr) + '.npy'))
        self.path = np.multiply(path, self.params['map_res']) # apply proper scaling
        self.init_pose, self.goal_pose = gt.get_init_and_goal_pose_full_path(path=self.path) # pose -> [x,y,psi]

        # Reset simulation, current vel, cmd_vel, cur_pos and orient and get local_goal_pos
        self.cur_pos, self.cur_orient_matrix = self.reset_webots()
        self.cur_vel = np.array([0.0, 0.0])
        self.cmd_vel = np.array([0.0, 0.0])
        self.local_goal_pos = gt.get_local_goal_pos(self.cur_pos, self.cur_orient_matrix, self.goal_pose)

        self.observation = self.get_obs()
        self.render(method='reset')

        return self.observation, {} # = info

    def step(self, action, teleop=False):
        # Updating previous values before updating
        self.prev_pos = self.cur_pos
        self.prev_observation = self.observation

        # Computing action projection
        if teleop == True:
            action = gt.get_teleop_action(self.keyboard)
            
        # NOTE: u must use an action wrapper for turning the action vector inot a cmd_vel
        self.cmd_vel = action

        # Inacting the action
        self.cur_vel = self.cmd_vel
        pos, orient = gt.compute_new_pose(self.params, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        self.robot_translation_field.setSFVec3f([pos[0], pos[1], pos[2]])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, orient])
        super().step(self.basic_timestep) # WEBOTS - Step()

        # Updating prev_pos, cur_pos and cur_orient, and getting new local goal
        self.cur_pos = np.array(self.robot_node.getPosition())
        self.cur_orient_matrix = np.array(self.robot_node.getOrientation())
        self.local_goal_pos = gt.get_local_goal_pos(self.cur_pos, self.cur_orient_matrix, self.goal_pose)

        self.observation = self.get_obs()
        self.reward = self.get_reward()
        done = self.get_done()
        self.render(method='step')

        return self.observation, self.reward, done, False, {} # last 2: truncated, info

    def get_obs(self):
        # Getting lidar data and converting to pointcloud
        super().step(self.basic_timestep) #NOTE: only after this timestep will the lidar data of the previous step be available
        observation = self.lidar_node.getRangeImage()
        # NOTE - heavy operation now only used for rendering purposes (TODO check what to do with it since it's already in ov wrapper)
        self.lidar_points = gt.lidar_to_point_cloud(self.params, self.precomputed_lidar_values, observation)

        return observation

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

        # When none of the done conditions are met
        return False

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

        # Return values
        cur_pos = np.array(self.robot_node.getPosition())
        cur_orient_matrix = np.array(self.robot_node.getOrientation())

        return cur_pos, cur_orient_matrix

    def set_render_mode(self, render_mode):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_init = True
        self.render_count = 0

    def render(self, method=None):
        if self.render_mode == None:
            return
        
        # Create initial plot or remove data to prepare for new data
        if self.render_init:
            self.render_init_plot(self.render_mode)
        else:
            self.render_remove_data(self.render_mode, method)
            
        # Add new data to the plot
        self.render_add_data(self.render_mode, method)
        
        # Plot graphs set flags and counter
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.render_init = False
        self.render_count += 1 # counter for reduced rendering (e.g. every 2nd step)
        
    def render_init_plot(self, render_mode, add_plots: int = 0):
        plt.ion()
        self.fig = plt.figure() 
        
        num_plots = add_plots + (2 if render_mode == 'full' else 1)
        num_rows = (num_plots + 1) // 2  # ensures at least two columns if more than one plot
        first_idx_add_plot = num_plots - 1
        
        gs = gridspec.GridSpec(num_rows, 2)
        
        # Create subplots dynamically and add to the list
        self.ax = []
        for i in range(num_plots):
            # For the last plot, if odd number, it should span both columns
            if i == num_plots - 1 and num_plots % 2 != 0:
                ax = self.fig.add_subplot(gs[i // 2, :])
            else:
                ax = self.fig.add_subplot(gs[i // 2, i % 2])
            self.ax.append(ax)
        
        if render_mode in ['position','full']:
            # ax[0] - Lidar data     #NOTE: plots footprint polygon only
            polygon = Polygon(self.params['polygon_coords'])
            patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.75, closed=True, facecolor='grey')
            self.ax[0].add_patch(patch)

            self.ax[0].set_xlim([-1.5,1.5])
            self.ax[0].set_ylim([-1.5,1.5])
            self.ax[0].set_xlabel('x [m]')
            self.ax[0].set_ylabel('y [m]')
            self.ax[0].set_aspect('equal')
            self.ax[0].grid()
            
        if render_mode in ['trajectory', 'full']:
            # ax[1] - Path, init pose and goal pose data
            self.ax[1].set_aspect('equal', adjustable='box')
            self.ax[1].set_xlabel('x [m]')
            self.ax[1].set_ylabel('y [m]')
        
        return first_idx_add_plot

    def render_remove_data(self, render_mode, method):
        if render_mode in ['position', 'full']:
            try:
                # ax[0] - Clear lidar data
                self.lidar_plot.remove()
                self.local_goal_plot.remove()
            except AttributeError:
                print("render(): removing position plot not possible because it doesn't exist yet")
                
        if render_mode in ['trajectory', 'full']:
            try:
                # ax[1] - Clear path and poses
                self.cur_pos_plot.remove()
                if method == 'reset':
                    self.grid_plot.remove()
                    self.path_plot.remove()
                    self.init_pose_plot.remove()
                    self.goal_pose_plot.remove()
            except AttributeError:
                print("render(): removing trajectory plot not possible because it doesn't exist yet")
                
    def render_add_data(self, render_mode, method):
        if render_mode in ['position', 'full']:
            # ax[0] - Lidar data
            self.lidar_plot = self.ax[0].scatter(self.lidar_points[:,0], self.lidar_points[:,1], alpha=1.0, c='black')
            self.local_goal_plot = self.ax[0].scatter(self.local_goal_pos[0], self.local_goal_pos[1], alpha=1.0, c='purple')
            
        if render_mode in ['trajectory', 'full']:
            # ax[1] - Path and poses
            self.cur_pos_plot = self.ax[1].scatter(self.cur_pos[0], self.cur_pos[1], c='blue', alpha=0.33)
            if method == 'reset':
                grid = np.load(os.path.join(self.grids_dir, 'grid_' + str(self.map_nr) + '.npy'))
                indices = np.argwhere(grid == 1)
                x, y = indices[:,0], indices[:,1]
                self.x_scaled, self.y_scaled = np.multiply(x, self.params['map_res']), np.multiply(y, self.params['map_res'])
                self.grid_plot = self.ax[1].scatter(self.x_scaled, self.y_scaled, marker='s', c='black')
                self.path_plot = self.ax[1].scatter(self.path[:,0], self.path[:,1], c='grey', alpha=0.5)
                self.init_pose_plot = self.ax[1].scatter(self.init_pose[0], self.init_pose[1], c='green')
                self.goal_pose_plot = self.ax[1].scatter(self.goal_pose[0], self.goal_pose[1], c='red')