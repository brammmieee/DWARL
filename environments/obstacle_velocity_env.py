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
import utils.obstacle_velocity_tools as ovt
import gymnasium as gym
from matplotlib.patches import Polygon as plt_polygon
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from matplotlib import gridspec

class CustomEnv(Supervisor, gym.Env):
    metadata = {"render_modes": ["full", "position", "velocity", "trajectory"], "render_fps": 4} # TODO: use fps in render function
    def __init__(self, render_mode=None, wb_open=True, wb_mode='training', reward_monitoring=False):
        # Directories
        self.package_dir = os.path.abspath(os.pardir)
        self.params_dir = os.path.join(self.package_dir, 'parameters')
        self.paths_dir = os.path.join(self.package_dir, 'paths')
        self.grids_dir = os.path.join(self.package_dir, 'grids')
        self.worlds_dir = os.path.join(self.package_dir, 'worlds')
        self.env_dir = os.path.join(self.package_dir, 'environments')

        # Parameters
        parameters_file = os.path.join(self.params_dir, 'parameters.yml')
        with open(parameters_file, "r") as file:
            self.params = yaml.safe_load(file)
        self.train_map_nr_list = at.read_pickle_file('train_map_nr_list', 'parameters')
        self.map_bounds_polygon = gt.compute_map_bound_polygon(self.params)

        # Precomputations
        self.precomputed = ovt.precomputations(self.params, visualize=False)

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_init = True
        self.render_count = 0

        # WEBOTS - Starting up a new webots sim and enabling fast modus
        if wb_open:
            world_file = os.path.join(self.worlds_dir, 'webots_world_file.wbt')

            # wb_process = Popen(["xvfb-run", "--auto-servernum", "webots", "--mode=fast", "--no-rendering", "--minimize", "--batch", world_file], stdout=PIPE)

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

        # WEBOTS - Static node references
        super().__init__() # Env class instance (self) inherits Supervisor's methods + Robot's methods
        self.basic_timestep = int(self.getBasicTimeStep())
        self.timestep = 2*self.basic_timestep #NOTE: basic timestep set 0.5*timestep for lidar update
        self.robot_node = self.getFromDef('ROBOT')
        self.root_node = self.getRoot() # root node (the nodes seen in the scene tree editor window are children of the root node)
        self.robot_translation_field = self.robot_node.getField('translation')
        self.robot_rotation_field = self.robot_node.getField('rotation')
        self.root_children_field = self.root_node.getField('children') # used for inserting map node

        # WEBOTS - Lidar sensor and keyboard
        self.lidar_node = self.getDevice('lidar')
        self.lidar_node.enable(int(self.getBasicTimeStep()))
        self.keyboard = Keyboard()

        # Space definitions
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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

        # Reward monitoring
        self.reward_monitoring = reward_monitoring
        if self.reward_monitoring == True:
            self.episode_nr = 1
            self.reward_matrix = []

    def reset(self, seed=None): #NOTE: removed -> options={"map_nr":1, "nominal_dist":1.0}
        # super().reset(seed=seed) # RNG seeding only done once (i.e. when value is not None) # TODO: check again

        # Reset the simulation
        self.simulationReset()
        super().step(self.basic_timestep) #NOTE: super prevents confusion with self.step() defined below

        # Setting simulation environment options
        train_map_nr_idx = random.randint(0, len(self.train_map_nr_list)-1)
        self.map_nr = self.train_map_nr_list[train_map_nr_idx]

        # Loading test path and apply proper scaling
        path = np.load(os.path.join(self.paths_dir, 'path_' + str(self.map_nr) + '.npy'))
        self.path = np.multiply(path, self.params['map_res'])

        # Reset init pose, goal pose
        self.init_pose, self.goal_pose = gt.get_init_and_goal_pose_full_path(path=self.path) #pose -> [x,y,psi]

        # Reset episode monitor and reward variables
        self.sim_time = 0.0
        self.stuck = False
        # self.waypoint_list = gt.get_waypoint_list(self.params, self.path)

        # WEBOTS - Loading and translating map into position
        self.root_children_field.importMFNodeFromString(position=-1, nodeString='DEF MAP ' + 'yaml_' + str(self.map_nr) + '{}')
        map_node = self.getFromDef('MAP')
        map_node_translation_field = map_node.getField('translation')
        map_node_translation_field.setSFVec3f([self.params['map_res']*self.params['map_width'], -(self.params['map_res']*self.params['map_width'])-3*self.params['map_res'], 0.0])
        super().step(self.basic_timestep)

        # WEBOTS - Positioning the robot at init_pos
        self.robot_translation_field.setSFVec3f([self.init_pose[0], self.init_pose[1], self.params['z_pos']]) #TODO: add z_pos to init_pose[2]
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, self.init_pose[3]])
        super().step(2*self.basic_timestep) #NOTE: 2 timesteps needed in order to succesfully set the init position

        # Reset current vel, action_proj, cur_pos and orient and get local_goal_pos
        self.cur_vel = np.array([0.0, 0.0])
        self.action_proj = np.array([0.0, 0.0])
        self.cur_pos = np.array(self.robot_node.getPosition())
        self.cur_orient_matrix = np.array(self.robot_node.getOrientation())
        self.local_goal_pos = gt.get_local_goal_pos(self.cur_pos, self.cur_orient_matrix, self.goal_pose)

        # Getting returned data
        self.observation = self.get_obs()

        # Render
        if self.render_mode != None:
            self.render(method='reset')

        # Reset reward list
        if self.reward_monitoring == True:
            self.reward_list = []
        
        # Set empty info
        info = {}

        return self.observation, info

    def step(self, action, teleop=False):
        # Updating previous values before updating
        self.prev_pos = self.cur_pos
        self.prev_observation = self.observation

        # Computing action projection
        if teleop == True:
            action = gt.get_teleop_action(self.keyboard)
        self.action_proj = self.get_action_projection(action)

        # Inacting the action
        self.cur_vel = self.action_proj
        pos, orient = gt.compute_new_pose(self.params, self.cur_pos, self.cur_orient_matrix, self.cur_vel)
        self.robot_translation_field.setSFVec3f([pos[0], pos[1], pos[2]])
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, orient])
        super().step(self.basic_timestep) # WEBOTS - Step()
        self.sim_time += self.timestep/1e3 # [s]

        # Updating prev_pos, cur_pos and cur_orient, and getting new local goal
        self.cur_pos = np.array(self.robot_node.getPosition())
        self.cur_orient_matrix = np.array(self.robot_node.getOrientation())
        self.local_goal_pos = gt.get_local_goal_pos(self.cur_pos, self.cur_orient_matrix, self.goal_pose)

        # Getting state (NOTE: depend on new cur_pos, cur_vel, sim_time)
        self.observation = self.get_obs()

        # Gettting reward (NOTE: depends on current observation)
        self.reward = self.get_reward() # TODO: pass done for time penalty
        done = self.monitor_episode() # TODO: add stuck monitor

        # Render (NOTE: depens on all the above)
        if self.render_mode != None:
            self.render(method='step')

        # Reward monitoring
        if self.reward_monitoring == True:
            self.reward_list.append(self.reward)
            if done:
                self.reward_matrix.append(self.reward_list)
                at.write_pickle_file(f'rewards_ep_{self.episode_nr}', os.path.join('training','rewards'), self.reward_matrix)
                self.episode_nr += 1

        # truncated dummy for adjusted gym -> gymanasium interface (note done=terminated)
        truncated = False

        info = {} # TODO: get_info()
        return self.observation, self.reward, done, truncated, info

    def get_obs(self):
        # Getting lidar data and converting to pointcloud
        super().step(self.basic_timestep) #NOTE: only after this timestep will the lidar data of the previous step be available
        lidar_range_image = self.lidar_node.getRangeImage()
        self.lidar_points = gt.lidar_to_point_cloud(self.params, self.precomputed, lidar_range_image)

        # Computing observation components
        self.vel_obs, self.vel_obs_mid = ovt.compute_velocity_obstacle(self.params, self.lidar_points, self.precomputed)
        self.dyn_win = ovt.compute_dynamic_window(self.params, self.cur_vel)
        self.goal_vel = ovt.compute_goal_vel_obs(self.params, self.local_goal_pos, self.cur_vel)

        observation = {"vel_obs": self.vel_obs_mid, "cur_vel": self.cur_vel, "dyn_win": self.dyn_win, "goal_vel": self.goal_vel}
        # print(f"{key} = {value}" for key, value in observation.items())

        return observation

    def get_reward(self):
        # Checking if @ goal with low velocity #NOTE: also used to terminate episode
        arrived_at_goal = False
        if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']): #TODO: pass from DONE instead of recompute
            arrived_at_goal = True #NOTE: arrived with low speed!!!

        # Goal/Progress reward
        if arrived_at_goal:
            r_goal = self.params['c_at_goal']
        else:
            r_goal = self.params['c_progr']*(np.linalg.norm(self.goal_pose[:2] - self.prev_pos[:2]) - np.linalg.norm(self.goal_pose[:2] - self.cur_pos[:2]))

        # Time penalty
        if arrived_at_goal == False:
            r_speed = -1
        else:
            r_speed = 0

        # Stuck penalty
        if self.stuck == True:
            r_stuck = -1.0
        else:
            r_stuck = 0.0

        # Total reward
        reward = r_goal + self.params['c_speed']*r_speed + self.params['c_stuck']*r_stuck

        return reward

    def monitor_episode(self): # Returns the value for done
        # Arrived at the goal
        if (np.linalg.norm(self.cur_pos[:2] - self.goal_pose[:2]) <= self.params['goal_tolerance']): #and (self.cur_vel[1] < self.params['v_goal_threshold']):
            return True

        # Getting stuck (i.e. there is no safe vel_cmd)
        if self.stuck:
            return True

        # Driving outside map limit
        cur_pos_point = Point(self.cur_pos[0], self.cur_pos[1])
        if not self.map_bounds_polygon.contains(cur_pos_point):
            if not self.map_bounds_polygon.boundary.contains(cur_pos_point):
                return True

        # When none of the done conditions are met
        else:
            return False

    def get_action_projection(self, action):
        # Project 2D action vector inside the dynamic window
        action_projection = np.array([
            self.cur_vel[0] + action[0]*self.params['alpha_max']*self.params['sample_time'],
            self.cur_vel[1] + action[1]*self.params['a_max']*self.params['sample_time'],
        ])

        # Clip action to the maximal velocity bounds
        low=np.array([self.params['omega_min'], self.params['v_min']])
        high=np.array([self.params['omega_max'], self.params['v_max']])
        action_projection = np.clip(a=action_projection, a_min=low, a_max=high)

        # If action inside vel_obs (and hence outside adm_vel_polygon), take closest safe point instead
        if not np.all(self.vel_obs == 0):
            adm_vel_polygon = Polygon(self.vel_obs)
            action_point = Point(action_projection[0], action_projection[1])
            if not adm_vel_polygon.contains(action_point):
                if not adm_vel_polygon.boundary.contains(action_point):
                    with np.errstate(invalid='ignore'):
                        closest_point, _ = nearest_points(adm_vel_polygon, action_point)
                    action_projection = np.array([closest_point.x, closest_point.y])
        else:
            action_projection = np.array([0.0, 0.0])
            self.stuck = True

        return action_projection

    def close(self):
        self.simulationQuit(0)

    def set_render_mode(self, render_mode):
        self.render_mode = render_mode

    def render(self, method=None): # NOTE: render_mode here for Sb3 env check
        if self.render_init == True:
            plt.ion()
            self.fig = plt.figure()
            gs = gridspec.GridSpec(2, 2)

            # Create the subplots
            self.ax = []
            self.ax.append(self.fig.add_subplot(gs[0, 0]))
            self.ax.append(self.fig.add_subplot(gs[0, 1]))
            self.ax.append(self.fig.add_subplot(gs[1, :]))

            # ax[0] - Lidar data     #NOTE: plots full discretisation (very heavy)
            # for i in range(len(self.precomputed['radii'])):
            #     for j in self.precomputed['rough_idx_matrix'][i]:
            #         point = self.precomputed['points'][i][j]
            #         self.ax[0].scatter(point[0], point[1], c='grey')

            #         polygon = self.precomputed['polygons'][i][j]
            #         patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.075, closed=True, facecolor='grey')
            #         self.ax[0].add_patch(patch)

            # ax[0] - Lidar data     #NOTE: plots footprint polygon only
            polygon = self.precomputed['polygons'][0][0]
            patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.75, closed=True, facecolor='grey')
            self.ax[0].add_patch(patch)

            self.ax[0].set_xlim([-1.5,1.5])
            self.ax[0].set_ylim([-1.5,1.5])
            self.ax[0].set_xlabel('x [m]')
            self.ax[0].set_ylabel('y [m]')
            self.ax[0].set_aspect('equal')
            self.ax[0].grid()

            # ax[2] - Observation, action and current velocity
            self.ax[2].plot([self.params['omega_max'], self.params['omega_max'], self.params['omega_min'], self.params['omega_min']],
                            [self.params['v_min'], self.params['v_max'], self.params['v_max'], self.params['v_min']], c='blue')
            self.ax[2].set_xlim([self.precomputed['omega_window_min'],self.precomputed['omega_window_max']])
            self.ax[2].set_ylim([self.precomputed['v_window_min'],self.precomputed['v_window_max']])
            self.ax[2].set_xlabel('$\omega$ [rad/s]')
            self.ax[2].set_ylabel('v [m/s]')
            self.ax[2].set_aspect('equal')
            self.ax[2].grid()

            # ax[1] - Path, init pose and goal pose data
            self.ax[1].set_aspect('equal', adjustable='box')
            self.ax[1].set_xlabel('x [m]')
            self.ax[1].set_ylabel('y [m]')

        elif self.render_init == False:
            if self.render_mode == 'position' or self.render_mode == 'full':
                try:
                    # ax[0] - Clear lidar data
                    self.lidar_plot.remove()
                    self.local_goal_plot.remove()
                except AttributeError:
                    print("render(): removing position plot not possible because it doesn't exist yet")

            if self.render_mode == 'velocity' or self.render_mode == 'full':
                try:
                    # ax[2] - Clear observation, action and current velocity
                    if self.vel_obs_mid.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
                        self.vel_obs_mid_plot.remove()
                    self.dyn_window_plot.remove()
                    self.goal_vel_plot.remove()
                    # for list_item in self.goal_vel_line_plot:
                    #     list_item.remove()
                    # self.vel_obs_patch_plot.remove()
                    self.dyn_win_patch_plot.remove()
                    self.cur_vel_plot.remove()
                    if method == 'step' and self.render_count >= 2:
                        self.action_proj_plot.remove()
                except AttributeError:
                    print("render(): removing velocity plot not possible because it doesn't exist yet")

            if self.render_mode == 'trajectory' or self.render_mode == 'full':
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

        if self.render_mode == 'position' or self.render_mode == 'full':
            # ax[0] - Lidar data
            self.lidar_plot = self.ax[0].scatter(self.lidar_points[:,0], self.lidar_points[:,1], alpha=1.0, c='black')
            self.local_goal_plot = self.ax[0].scatter(self.local_goal_pos[0], self.local_goal_pos[1], alpha=1.0, c='purple')

        if self.render_mode == 'velocity' or self.render_mode == 'full':
            # ax[2] - Observation, action and current velocity
            if self.vel_obs_mid.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
                self.vel_obs_mid_plot = self.ax[2].scatter(self.vel_obs_mid[:,0], self.vel_obs_mid[:,1], c='black')
            self.dyn_window_plot = self.ax[2].scatter(self.dyn_win[:,0], self.dyn_win[:,1], c='blue')
            self.goal_vel_plot = self.ax[2].scatter(self.observation['goal_vel'][0], self.observation['goal_vel'][1], c='purple')
            # self.goal_vel_line_plot = self.ax[2].plot([0.0, self.goal_vel_ub[0]], [0.0, self.goal_vel_ub[1]], c='purple')
            # if self.vel_obs.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
            #     vel_obs_patch = plt_polygon(self.vel_obs, alpha=0.17, closed=True, facecolor='black')
            dyn_win_patch = plt_polygon(self.dyn_win, alpha=0.17, closed=True, facecolor='blue')
            # if self.vel_obs.shape != (0,): # protect against invalid acces (when no obstacles present and vel_obs is empty)
            #     self.vel_obs_patch_plot = self.ax[2].add_patch(vel_obs_patch)
            self.dyn_win_patch_plot = self.ax[2].add_patch(dyn_win_patch)
            self.cur_vel_plot = self.ax[2].scatter(self.cur_vel[0], self.cur_vel[1], c='blue', marker='+', alpha=0.99)
            if method == 'step':
                self.action_proj_plot = self.ax[2].scatter(self.action_proj[0], self.action_proj[1], color='green', alpha=0.99)

        if self.render_mode == 'trajectory' or self.render_mode == 'full':
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

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.render_init = False
        self.render_count += 1
