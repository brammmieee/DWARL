import os
import sys
import json
import numpy as np
import random
from typing import Tuple, List, Dict
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.strtree import STRtree

from utils.admin_tools import find_file, load_parameters

def chain_wrappers(env, wrapper_classes):
    for wrapper_class in wrapper_classes:
        env=wrapper_class(env)
        
    return env

def killall_webots():
    command = "ps aux | grep webots | grep -v grep | awk '{print $2}' | xargs -r kill"
    os.system(command)

def get_teleop_action(keyboard):
    params = load_parameters('base_parameters.yaml')

    key = float(keyboard.getKey())
    if key == 315: # up arrow
        action = np.array([0.0, params['v_max']])
    elif key == 317: # down arrow
        action = np.array([0.0, params['v_min']])
    elif key == 314: # left array
        action = np.array([params['omega_min'], 0.0]) #.00001 to not get stuck on upper bounds
    elif key == 316: # right arrow
        action = np.array([params['omega_max'], 0.0]) #.00001 to not get stuck on upper bounds
    else:
        action = np.array([0.0, 0.0])
    return action

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
        return init_pose, goal_pose

    elif mode == 'random':
        if 'nominal_distance' == None:
            raise ValueError("Missing 'nominal_distance' parameter for init_and_goal_pose_mode: 'random'.")
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

def get_cmd_vel(robot_node):
    webots_vel = robot_node.getVelocity()
    ang_vel = (-1)*webots_vel[-1] #NOTE: times (-1) because clockwise rotation is taken as the positve direction
    lin_vel = np.sqrt(webots_vel[0]**2 + webots_vel[1]**2) # in plane global velocities (x and y) to forward vel
    return np.array([ang_vel, lin_vel])

def precompute_lidar_values(num_lidar_rays):
    lidar_delta_psi = (2 * np.pi) / num_lidar_rays
    lidar_angles = np.arange(0, 2 * np.pi, lidar_delta_psi)
    lidar_cosines = np.cos(lidar_angles)
    lidar_sines = np.sin(lidar_angles)
    
    return {
        "lidar_cosines": lidar_cosines,
        "lidar_sines": lidar_sines,
    }

def lidar_to_point_cloud(parameters, precomputed, lidar_range_image):
    # NOTE: Webots axis conventions w.r.t. the conventions of this package
    with np.errstate(invalid='ignore'): # prevent runtime errors
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

def sample_training_test_map_nrs(first_map_idx: int, last_map_idx: int, training_ratio: float) -> Tuple[List[int], List[int]]:
    """
    Splits a range of numbers into training and testing lists based on a given ratio.
    
    Args:
    first_map_idx (int): The first index of the map number range.
    last_map_idx (int): The last index of the map number range.
    training_ratio (float): The fraction of the total range to be used for training.
    
    Returns:
    Tuple[List[int], List[int]]: A tuple containing two lists:
        - The first list contains the training indices.
        - The second list contains the testing indices.
    """
    all_numbers = list(range(first_map_idx, last_map_idx + 1))
    training_list_size = int(len(all_numbers) * training_ratio)
    training_list = random.sample(all_numbers, training_list_size)
    testing_list = [x for x in all_numbers if x not in training_list]
    
    return training_list, testing_list

def create_level_dictionary(input_list: List[int], num_levels: int, total_maps: int) -> Dict[str, List[int]]:
    """
    Organizes a list of map indices into a dictionary based on specified levels.
    
    Args:
    input_list (List[int]): The list of integers (map indices) to be organized.
    num_levels (int): The number of levels to divide the maps into.
    total_maps (int): The total number of maps.
    
    Returns:
    Dict[str, List[int]]: A dictionary where each key corresponds to a level ("lvl x") and each value is a list of integers assigned to that level.
    """
    level_dict = {}
    maps_per_level = total_maps // num_levels  # Assumes an equal distribution of maps across levels
    for i in range(1, num_levels + 1):
        lower_bound = (i - 1) * maps_per_level
        if i < num_levels:
            upper_bound = i * maps_per_level
        else:
            upper_bound = total_maps  # Ensure the last level includes any remaining maps due to integer division

        level_key = f'lvl {i}'
        level_values = [num for num in input_list if lower_bound <= num < upper_bound]
        level_dict[level_key] = level_values

    return level_dict

def sample_maps(train_map_nr_dict, test_map_nr_dict, maps_per_level, lowest_level, highest_level):
    """
    Sample maps from level-based dictionaries for training and testing.

    Args:
    train_map_nr_dict (dict): A dictionary containing training maps categorized by levels.
    test_map_nr_dict (dict): A dictionary containing testing maps categorized by levels.
    maps_per_level (int): Number of maps to sample per level.
    lowest_level (int): Lowest level to consider.
    highest_level (int): Highest level to consider.

    Returns:
    tuple: Two lists containing the sampled training and testing map numbers.
    """
    train_map_nr_list = []
    test_map_nr_list = []
    
    for i, map_nr_dict in enumerate([train_map_nr_dict, test_map_nr_dict]):
        for lvl in range(lowest_level, highest_level + 1):
            level_key = f'lvl {lvl}'
            map_nr_list = map_nr_dict.get(level_key, [])
            if i == 0:  # Training maps
                train_map_nr_list.extend(sorted(random.sample(map_nr_list, maps_per_level)))
            else:  # Testing maps
                test_map_nr_list.extend(sorted(random.sample(map_nr_list, maps_per_level)))
                
    return train_map_nr_list, test_map_nr_list

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

def update_protos(config_file_name, package_dir=os.path.abspath(os.pardir)):
    """
    Update the proto files based on the provided configuration.

    Args:
    config_file_name (str): The name of the configuration file.
    package_dir (str, optional): The directory of the config file. Defaults to the parent directory of the current directory.

    Raises:
    FileNotFoundError: If the config file or template proto file is not found.
    """
    try:
        # Load the config file
        config_folder = os.path.join(package_dir, 'parameters', 'proto_configs')
        config_file_path = find_file(filename=config_file_name, start_dir=config_folder)
        with open(config_file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found.")

    try:
        template_proto_file_name = config['template_name']
        output_proto_file_name = config['output_name']
        
        # Load the template proto file
        proto_folder = os.path.join(package_dir, 'resources', 'protos')
        template_proto_file_path = find_file(filename=template_proto_file_name, start_dir=proto_folder)
        with open(template_proto_file_path, 'r') as file:
            template_proto_content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError("Template proto file not found.")

    # Replace placeholders in the content with values from the config
    output_proto_content = replace_placeholders(template_proto_content, config['proto_substitutions'])

    # Write the updated content to the output file
    output_proto_file_path = template_proto_file_path.replace(template_proto_file_name, output_proto_file_name)
    with open(output_proto_file_path, 'w') as file:
        file.write(output_proto_content)