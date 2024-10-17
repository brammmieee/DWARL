from controller import Supervisor, Keyboard
from shapely.affinity import translate, rotate
from shapely.geometry import Point, Polygon
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.strtree import STRtree
from subprocess import Popen, PIPE
import math as m
import numpy as np
import os
import yaml

class WebotsEnv(Supervisor):
    @staticmethod
    def killall():
        command = "ps aux | grep webots | grep -v grep | awk '{print $2}' | xargs -r kill"
        os.system(command)

    @staticmethod
    def open_world(cfg, path_to_worlds):       
        # Create Webots command with specified mode and world file
        world_file = os.path.join(path_to_worlds, 'webots_world_file.wbt')
        cmd = ['webots','--extern-urls', '--no-rendering', f'--mode={cfg.mode}', world_file]

        # Open Webots
        wb_process = Popen(cmd, stdout=PIPE)

        # Set the environment variable for the controller to connect to the supervisor
        output = wb_process.stdout.readline().decode("utf-8")
        ipc_prefix = 'ipc://'
        start_index = output.find(ipc_prefix)
        port_nr = output[start_index + len(ipc_prefix):].split('/')[0]
        os.environ["WEBOTS_CONTROLLER_URL"] = ipc_prefix + str(port_nr)
        
    def __init__(self, cfg, path_to_worlds):
        super().__init__()

        self.basic_timestep = int(self.getBasicTimeStep())
        self.timestep = 2*self.basic_timestep #NOTE: basic timestep set 0.5*timestep for lidar update
        
        # Static node references
        self.robot_node = self.getFromDef('ROBOT')
        self.root_node = self.getRoot() # root node (the nodes seen in the Webots scene tree editor window are children of the root node)
        self.robot_translation_field = self.robot_node.getField('translation')
        self.robot_rotation_field = self.robot_node.getField('rotation')
        self.root_children_field = self.root_node.getField('children') # used for inserting map node

        # Lidar sensor and keyboard
        self.lidar_node = self.getDevice('lidar')
        self.lidar_node.enable(int(self.getBasicTimeStep()))
        self.keyboard = Keyboard()

        # Open the world
        self.open_world(cfg, path_to_worlds)
    
    def reset(self, map_nr):
        self.simulationReset()
        super().step(self.basic_timestep) # super prevents confusion with self.step() defined below
        self.reset_map(map_nr)
        self.reset_robot()

    def reset_map(self, map_nr):
        # Loading and translating map into position
        self.root_children_field.importMFNodeFromString(position=-1, nodeString='DEF MAP ' + 'yaml_' + str(self.map_nr) + '{}')
        map_node = self.getFromDef('MAP')
        map_node_translation_field = map_node.getField('translation')
        map_node_translation_field.setSFVec3f([self.params['map_res']*self.params['map_width'], -(self.params['map_res']*self.params['map_width'])-3*self.params['map_res'], 0.0])
        super().step(self.basic_timestep)

    def reset_robot(self):
        # Positioning the robot at init_pos
        self.robot_translation_field.setSFVec3f([self.init_pose[0], self.init_pose[1], self.params['z_pos']]) #TODO: add z_pos to init_pose[2]
        self.robot_rotation_field.setSFRotation([0.0, 0.0, 1.0, self.init_pose[3]])
        super().step(2*self.basic_timestep) #NOTE: 2 timesteps needed in order to succesfully set the init position
    
    def step(self, new_position, new_orientation):
        self.robot_translation_field.setSFVec3f([
            new_position[0], 
            new_position[1], 
            new_position[2]
        ])
        self.robot_rotation_field.setSFRotation([
            0.0,
            0.0,
            1.0,
            new_orientation
        ])
        super().step(self.basic_timestep)
        super().step(self.basic_timestep) #NOTE: only after this timestep will the lidar data of the previous step be available

    def close_webots(self):
        self.simulationQuit(0)
        self.killall()

    @property
    def robot_position(self):
        return np.array(self.robot_node.getPosition())

    @property
    def robot_orientation(self):
        return np.array(self.robot_node.getOrientation())

class map_dataloader:
    def __init__(self, paths):
        self.paths = paths # paths as in system paths

    def load_dataset(self, dataset_name):
        dataset = {}
        path_to_datasets = self.paths.resources.datasets
        map_configs_path = self.paths.resources.map_configs
        path_to_grids = self.paths.resources.grids
        path_to_paths = self.paths.resources.paths
        path_to_protos = self.paths.resources.protos

        with open(os.path.join(path_to_datasets, dataset_name + '.yaml'), 'r') as file:
            dataset = yaml.safe_load(file)
        
        # at.load_from_json('train_map_nr_list.json', os.path.join(self.params_dir, 'map_nrs'))
        return dataset
    
    def 
    
def check_collision(collision_tree, footprint_glob):
    # Use the STRTree to find any intersecting boxes
    result = collision_tree.query(footprint_glob, predicate='intersects')
    collision_detected = len(result) > 0

    if collision_detected:
        return True
    
    return collision_detected

def get_global_footprint_location(cur_pos, cur_orient_matrix, polygon_coords):
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

def get_local_goal_pos(cur_pos, cur_orient_matrix, goal_pose): #TODO: remove 3rd dimension for efficiency
    '''
    Get the goal position in the local frame of reference of the robot.
    '''
    # Convert inputs to numpy arrays with the right dimensions
    cur_pos = np.array([cur_pos[0], cur_pos[1], 1])
    goal_pos = np.array([goal_pose[0], goal_pose[1], 1]) #NOTE: pose vs pos here
    cur_orient_matrix = np.array([[cur_orient_matrix[0], cur_orient_matrix[1], cur_orient_matrix[2]],
                                  [cur_orient_matrix[3], cur_orient_matrix[4], cur_orient_matrix[5]],
                                  [cur_orient_matrix[6], cur_orient_matrix[7], cur_orient_matrix[8]]])
    
    # Calculate the translation vector
    translation = cur_pos - goal_pos

    #Compensate for different axis convention compared to Webots
    rot_z = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) 
    cur_orient_matrix = np.dot(rot_z, cur_orient_matrix) # 90-degree rotation around the z-axis
    
    # Calculate the goal position in the local frame of reference
    local_goal_pos = np.dot(cur_orient_matrix.T, translation)
    return local_goal_pos

def compute_new_pose(parameters, cur_pos, cur_orient_matrix, cur_vel):
    '''
    Kinematic compution of the new pose based on the current pose and velocity.
    '''
    # Computing the orientation
    psi = (np.arctan2(cur_orient_matrix[3], cur_orient_matrix[0])) % (2*np.pi)
    local_rotation = cur_vel[0]*parameters['sample_time']
    global_rotation = -local_rotation #NOTE: minus to account for our conventions
    orientation = psi + global_rotation 
    
    # Computing the position
    distance = cur_vel[1]*parameters['sample_time']
    if abs(cur_vel[0]) > 1e-10 and abs(cur_vel[1]) > 1e-10: # Curve
        radius = cur_vel[1]/cur_vel[0]
        gamma = distance/radius
        local_translation = np.array([radius - radius*np.cos(gamma), radius*np.sin(gamma)])
    elif abs(cur_vel[0]) < 1e-10 and abs(cur_vel[1]) > 1e-10: # Pure translation
        local_translation = np.array([0.0, distance])
    else: # Pure rotation or standstill
        local_translation = np.array([0.0, 0.0])
    global_translation = np.array([
        np.cos(psi)*local_translation[1] - np.sin(psi)*local_translation[0], 
        np.sin(psi)*local_translation[1] + np.cos(psi)*local_translation[0]
        ])
    position = np.array([
        cur_pos[0] + global_translation[0], 
        cur_pos[1] + global_translation[1], 
        parameters['z_pos']
        ])

    return position, orientation

def calculate_orientation(p1, p2):
    """ Calculate orientation angle from point p1 to p2. """
    align = p2 - p1
    return np.arctan2(align[1], align[0])

def create_pose(point, orientation=0.0):
    """ Create a pose from a point and an orientation. """
    return np.array([point[0], point[1], 0.0, orientation])

def get_init_and_goal_poses(path, parameters=None):
    """
    General function to get initial and goal poses on a path.
    
    Args:
        path: List or array of points defining the path.
        mode: Type of mode to determine poses ('random', 'full_path', 'random_distance').
        parameters: Dictionary containing parameters like nominal distance, sign, etc.

    Returns:
        A tuple of numpy arrays (init_pose, goal_pose).
    """
    mode = parameters.get('init_and_goal_pose_mode')
    if mode is None:
        raise ValueError("Missing 'init_and_goal_pose_mode' parameter.")

    if mode == 'full_path':
        init_pose = create_pose(path[0], calculate_orientation(path[0], path[min(2, len(path) - 1)]))
        goal_pose = create_pose(path[-1])
        direction = 1
        return init_pose, goal_pose, 1

    elif mode == 'random':
        nom_dist = parameters['nominal_distance']
        sign = np.random.choice([-1, 1])
        rand_index = np.random.randint(0, len(path))

        i = rand_index
        dist_traveled = 0
        while 0 <= i + sign < len(path):
            segment_dist = np.linalg.norm(path[i + sign] - path[i])
            if dist_traveled + segment_dist >= nom_dist:
                init_pose = create_pose(path[rand_index], calculate_orientation(path[rand_index], path[rand_index + sign]))
                goal_pose = create_pose(path[i], calculate_orientation(path[i], path[i + sign]))
                direction = sign
                return init_pose, goal_pose, direction
            dist_traveled += segment_dist
            i += sign

        # Recursive call fallback for cases when distance is not met
        sign *= -1
        return get_init_and_goal_poses(path, parameters={'init_and_goal_pose_mode': mode, 'nominal_distance': nom_dist, 'sign': sign, 'index': rand_index})

    elif mode == 'random_distance':
        init_path_index = np.random.randint(0, len(path) - 1)
        goal_path_index = np.random.randint(0, len(path) - 1)
        if np.linalg.norm(path[init_path_index] - path[goal_path_index]) < parameters['min_init2goal_dist']:
            return get_init_and_goal_poses(path, parameters=parameters)
        else:
            init_pose = create_pose(path[init_path_index], calculate_orientation(path[init_path_index], path[goal_path_index]))
            goal_pose = create_pose(path[goal_path_index], calculate_orientation(path[goal_path_index], path[init_path_index]))
            direction = 1 if init_path_index < goal_path_index else -1
            return init_pose, goal_pose, direction
        
    else:
        raise ValueError("Unsupported mode provided.")

##### NOTE!!! are we sure the get velocity call actually gives us a velocity? #####
def get_cmd_vel(robot_node):
    webots_vel = robot_node.getVelocity()
    ang_vel = (-1)*webots_vel[-1] #NOTE: times (-1) because clockwise rotation is taken as the positve direction
    lin_vel = np.sqrt(webots_vel[0]**2 + webots_vel[1]**2) # in plane global velocities (x and y) to forward vel
    return np.array([ang_vel, lin_vel])

def apply_kinematic_constraints(params, cur_vel, target_vel):
    omega_max = params['omega_max']
    omega_min = params['omega_min']
    alpha_max = params['alpha_max']
    alpha_min = params['alpha_min']
    a_max = params['a_max']
    a_min = params['a_min']
    v_max = params['v_max']
    v_min = params['v_min']
    dt = params['sample_time']

    domega = target_vel[0] - cur_vel[0]
    domega_clipped = np.clip(domega, alpha_min*dt, alpha_max*dt) 
    omega_clipped = np.clip((cur_vel[0] + domega_clipped), omega_min, omega_max)

    dv = target_vel[1] - cur_vel[1]
    dv_clipped = np.clip(dv, a_min*dt, a_max*dt)
    v = np.clip((cur_vel[1] + dv_clipped), v_min, v_max)

    return np.array([omega_clipped, v])

def precompute_lidar_values(num_lidar_rays):
    lidar_delta_psi = (2 * np.pi) / num_lidar_rays
    lidar_angles = np.arange(0, 2 * np.pi, lidar_delta_psi)
    lidar_cosines = np.cos(lidar_angles)
    lidar_sines = np.sin(lidar_angles)
    
    return {
        "lidar_cosines": lidar_cosines,
        "lidar_sines": lidar_sines,
    }

def lidar_to_point_cloud(parameters, precomputed, lidar_range_image, replace_value=0):
    # NOTE: Webots axis conventions w.r.t. the conventions of this package
    lidar_range_image = np.array(lidar_range_image)
    lidar_range_image[np.isinf(lidar_range_image)] = replace_value
    lidar_range_image[np.isnan(lidar_range_image)] = replace_value

    lidar_points = np.column_stack(( 
        lidar_range_image*-precomputed['lidar_sines'], #NOTE: minus because lidar type was set to fixed
        lidar_range_image*-precomputed['lidar_cosines']
    ))
    # Remove rows (i.e. points) that have np.inf or np.nan in as either x or y value (causes issues in search tree)
    invalid_indices = np.logical_or(np.isinf(lidar_points).any(axis=1), np.isnan(lidar_points).any(axis=1))
    lidar_points = lidar_points[~invalid_indices]

    # Add lidar position offset
    lidar_points[:,1] += parameters['lidar_y_pos']
    return lidar_points

def precompute_collision_detection(gridmap, resolution):
    """
    Precomputes the collision detection data structure.

    Args:
    gridmap (list of list of int): Occupancy grid (2D array) where 1 indicates an occupied cell and 0 an empty one.
    resolution (float): The size of each grid cell in meters.

    Returns:
    shapely.strtree.STRTree: The constructed STRTree.
    """
    occupied_boxes = []
    
    # Create a shapely box for each occupied cell in the gridmap
    for x in range(len(gridmap)):
        for y in range(len(gridmap[0])):
            if gridmap[x][y] == 1:
                # Convert grid indices to spatial coordinates based on the resolution
                min_x = x * resolution
                min_y = y * resolution
                max_x = min_x + resolution
                max_y = min_y + resolution
                # Create a box for the occupied cell
                occupied_boxes.append(box(min_x, min_y, max_x, max_y))
    
    # Create an STRTree from all occupied boxes
    return STRtree(occupied_boxes)

def compute_map_bound_polygon(parameters):
    map_res = parameters['map_res']
    map_width = parameters['map_width']
    map_bound_buffer = parameters['map_bound_buffer']

    map_bounds = [
        [0.0, 0.0], 
        [(map_width*map_res - map_res), 0.0], 
        [(map_width*map_res - map_res), (map_width*map_res - map_res)], 
        [0.0, (map_width*map_res - map_res)]
    ]

    map_bound_polygon = Polygon(map_bounds)
    buffered_map_bound_polygon = map_bound_polygon.buffer(map_bound_buffer)
    
    return Polygon(buffered_map_bound_polygon)

def get_polygon_bounds(polygon: Polygon):
    x, y = polygon.exterior.coords.xy
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    return min_x, max_x, min_y, max_y

def get_waypoint_list(parameters, path):
    wp_tolerance = parameters['wp_tolerance']

    i = 1
    dist_traveled = 0
    waypoint_list = []

    while (i < len(path)):
        segment_dist = np.linalg.norm(path[i] - path[i-1])
        dist_traveled += segment_dist
        
        if dist_traveled > wp_tolerance:
            waypoint_list.append(path[i])
            dist_traveled = 0 # reset distance

        i += 1

    return waypoint_list

def replace_placeholders(content, substitutions):
    """
    Replace placeholders in the content with values from the substitutions dictionary.

    Args:
    content (str): The content string to be modified.
    substitutions (dict): A dictionary containing the placeholder-value pairs.

    Returns:
    str: The modified content string.
    """
    for key, value in substitutions.items():
        placeholder = f"{{{{{key}}}}}"
        if placeholder not in content:
            content = content.replace(placeholder, str(value))
            print(f"Replacing '{placeholder}' with '{value}' in the proto file.")
        content = content.replace(placeholder, str(value))
    return content

def update_protos(cfg, path_to_proto):
    """
    Update the proto files based on the provided configuration.

    Args:
    config_file_name (str): The name of the configuration file.
    package_dir (str, optional): The directory of the config file. Defaults to the parent directory of the current directory.

    Raises:
    FileNotFoundError: If the config file or template proto file is not found.
    """
    try:
        template_proto_file_name = cfg.template_name
        output_proto_file_name = cfg.output_name
        
        # Load the template proto file
        template_proto_file_path = find_file(
            filename=template_proto_file_name, 
            start_dir=path_to_proto
        )
        with open(template_proto_file_path, 'r') as file:
            template_proto_content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError("Template proto file not found.")

    # Replace placeholders in the content with values from the config
    output_proto_content = replace_placeholders(
        template_proto_content, 
        cfg.substitutions
    )

    # Write the updated content to the output file
    output_proto_file_path = template_proto_file_path.replace(template_proto_file_name, output_proto_file_name)
    with open(output_proto_file_path, 'w') as file:
        file.write(output_proto_content)