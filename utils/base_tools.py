import os
import json
import numpy as np
import random
from typing import Tuple, List, Dict
from shapely.geometry import Polygon

from utils.admin_tools import find_file

import warnings
warnings.simplefilter("error", RuntimeWarning)

def get_teleop_action(keyboard):
    key = float(keyboard.getKey())
    if key == 315.0: # up arrow
        action = np.array([0.0, 1.0])
    elif key == 317: # down arrow
        action = np.array([0.0, -1.0])
    elif key == 314: # left array
        action = np.array([-1.0, 0.0]) #.00001 to not get stuck on upper bounds
    elif key == 316: # right arrow
        action = np.array([1.0, 0.0]) #.00001 to not get stuck on upper bounds
    else:
        action = np.array([0.0, 0.0])
    return action

def get_local_goal_pos(cur_pos, cur_orient_matrix, goal_pose): #TODO: remove 3rd dimension for efficiency
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

def get_init_and_goal_pose(path, nom_dist=1.0, sign=None, index=None): 
    # (NOTE: assumes nom_dist < 0.5 path_len)
    # Choose a random initial location on the path
    if index == None: # non-recursive call
        rand_index = np.random.randint(0, len(path))
    else:
        rand_index = index

    # Choose a random direction sign at the initial position
    if sign == None: # non-recursive call
        sign = np.random.choice([-1, 1])    

    # Loop through the path points in sign direction keeping track of the traveled distance
    i = rand_index
    dist_traveled = 0
    while (i < len(path)) and (i >= 0):
        if (i + sign < len(path)) and (i + sign >= 0):
            segment_dist = np.linalg.norm(path[i+sign] - path[i])
            if (dist_traveled + segment_dist < nom_dist):
                dist_traveled += segment_dist
                i += sign 
            elif (dist_traveled + segment_dist >= nom_dist):
                # init_pose
                init_pose = np.array([path[rand_index][0], path[rand_index][1], 0.0, 0.0])  # x, y, z, rotation
                init_align = path[rand_index + sign] - init_pose[:2]
                init_pose[3] = np.arctan2(init_align[1], init_align[0])   
                # goal_pose
                goal_pose = path[i]
                goal_align = goal_pose - init_pose[:2]
                goal_pose = np.array([goal_pose[0], goal_pose[1], 0.0, np.arctan2(goal_align[1], goal_align[0])])
                return init_pose, goal_pose
        else: # Recursive call with opposite direction (NOTE: assumes nom_dist < 0.5 path_len)
            return get_init_and_goal_pose(path, nom_dist, sign=(-1)*sign, index=rand_index)
        
def get_init_and_goal_pose_full_path(path):
    # Randomly invert the path 

    # Init pose as first path pose
    init_pose = np.array([path[0][0], path[0][1], 0.0, 0.0])  # x, y, z, rotation
    init_align = path[2] - init_pose[:2]
    init_pose[3] = np.arctan2(init_align[1], init_align[0]) 

    # Goal pose as final path POSition
    goal_index = int(len(path)-1)
    goal_pose = np.array([path[goal_index][0], path[goal_index][1], 0.0, 0.0]) #NOTE: goal_pose only holds positional information -> the rest is not used to also not set here!

    return init_pose, goal_pose

def get_init_and_goal_pose_random_nom_dist(parameters, path):
    # Initialising init and goal pose
    init_pose = np.array([0.0, 0.0, 0.0, 0.0]) # x,y,z,orient_angle
    goal_pose = np.array([0.0, 0.0, 0.0, 0.0])

    # Getting random path indices
    init_path_index = np.random.randint(0, len(path)-1)
    goal_path_index = np.random.randint(0, len(path)-1)

    # Setting random positions
    init_pose[:2] = path[init_path_index]
    goal_pose[:2] = path[goal_path_index]

    # Checking if minimal dist constrained is met
    if np.linalg.norm(init_pose[:2] - goal_pose[:2]) >= parameters['min_init2goal_dist']:
        # Seting init pose orientation (NOTE: goal pose orient is not necessary)
        if goal_path_index > init_path_index:
            init_align = path[init_path_index + 1] - init_pose[:2]
        elif goal_path_index < init_path_index:
            init_align = path[init_path_index - 1] - init_pose[:2]
        else:
            print('get_init_goal_pose..(): index fault')
        init_pose[3] = np.arctan2(init_align[1], init_align[0])
        return init_pose, goal_pose
    else:        
        return get_init_and_goal_pose_random_nom_dist(parameters, path)

def get_cmd_vel(robot_node):
    webots_vel = robot_node.getVelocity()
    ang_vel = (-1)*webots_vel[-1] #NOTE: times (-1) because clockwise rotation is taken as the positve direction
    lin_vel = np.sqrt(webots_vel[0]**2 + webots_vel[1]**2) # in plane global velocities (x and y) to forward vel
    return np.array([ang_vel, lin_vel])

def precompute_lidar_values(parameters):
    lidar_delta_psi = (2 * np.pi) / parameters['num_lidar_rays']
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
    output_proto_content = replace_placeholders(template_proto_content, config['substitutions'])

    # Write the updated content to the output file
    output_proto_file_path = template_proto_file_path.replace(template_proto_file_name, output_proto_file_name)
    with open(output_proto_file_path, 'w') as file:
        file.write(output_proto_content)