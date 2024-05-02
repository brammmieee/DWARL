import numpy as np
import math
from shapely.geometry import Polygon

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