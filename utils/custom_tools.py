import os
import numpy as np
import pickle
import math
import random
import matplotlib.pyplot as plt
import datetime
from matplotlib.patches import Polygon as plt_polygon
from shapely.geometry import Point, MultiPoint, MultiPolygon, Polygon, LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.affinity import translate, rotate
from scipy.spatial import KDTree

import warnings
warnings.simplefilter("error", RuntimeWarning)

def write_pickle_file(file_name, file_dir, value):
    # NOTE: file dir is relative to package folder
    package_dir = os.path.abspath(os.pardir)
    pickle_file_path = os.path.join(package_dir, file_dir, file_name + '.pickle')
    with open(pickle_file_path, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    
def read_pickle_file(file_name, file_dir):
    package_dir = os.path.abspath(os.pardir)
    pickle_file_path = os.path.join(package_dir, file_dir, file_name + '.pickle')
    with open(pickle_file_path, "rb") as file:
        pickle_file = pickle.load(file)
        return pickle_file
    
def create_day_folder(file_dir):
    # NOTE: file_dir is the dir relative to the 
    package_dir = os.path.abspath(os.pardir)
    current_date = datetime.datetime.now()
    day_folder = current_date.strftime('%d')  # Format: DD
    directory = os.path.join(package_dir, file_dir, day_folder)

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_name_with_date(test_nr_today, comment=''):
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%B%d")
    return f'{formatted_date}V{str(test_nr_today)}{comment}'

def get_file_name_with_date_testing(test_nr_today, comment):
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%B_%d")
    return f'{formatted_date}_V{str(test_nr_today)}_{comment}'

def sample_training_test_map_nrs(range_start, range_end, training_ratio):
    all_numbers = list(range(range_start, range_end + 1))
    print('all nums', all_numbers)

    training_list_size = int(len(all_numbers)*training_ratio)
    print('training_list_size', training_list_size)

    training_list = random.sample(all_numbers, training_list_size)
    testing_list = [x for x in all_numbers if x not in training_list]
    
    return training_list, testing_list

def sample_training_test_map_nrs2(map_list, training_ratio):
    training_list_size = math.floor(len(map_list)*training_ratio)
    print('training_list_size', training_list_size)

    training_list = random.sample(map_list, training_list_size)
    testing_list = [x for x in map_list if x not in training_list]
    
    return training_list, testing_list

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

def precomputations(parameters, visualize=False):
    # =================== Velocity Window ====================== #
    omega_window_max = np.sqrt(2*parameters['alpha_max']*parameters['max_lidar_range'])
    omega_window_min =  -1*omega_window_max
    v_window_max = np.sqrt(2*parameters['a_max']*parameters['max_lidar_range'])
    v_window_min = 0.0

    # =================== Velocity Obstacle ==================== #
    # Plotting discretisation:
    if visualize: 
        fig, ax = plt.subplots()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    
    # determine the maximum dist based on the adm vels
    dist_max = max(v_window_max**2/(2*abs(parameters['a_min'])), 
                   omega_window_max**2/(2*abs(parameters['alpha_min']))) #NOTE: improve efficiency even more by taking min based on the quadrant of r (complex)
     
    # Calculating radii and radii_ad (additional)    
    if parameters['num_radii']%2 != 0:
        print('WARNING: uneven values for the number of radii causes errors in calculating the tangent!')
        print('please choose an even amount')
    
    if not parameters['new_discretisation']:
        ## Original discretisation based on gamma max
        gammas_rad = np.linspace(1.0e-15, parameters['gamma_rad_max'], parameters['num_radii']) #NOTE: the np.pi*.. is based on the visualisation -> should be no room for obstacles next to the robot
        radii_int_pos = np.sort(dist_max/gammas_rad)
        print('Precompute(): Employing OLD discretisation method')
    else:
        ## New 1: based on gamma
        gammas_rad = []
        for x in range(parameters['num_radii']):
            if x < 1:
                gammas_rad.append(0.001)
            else:
                gammas_rad.append(0.2+(x/(2*np.pi))**3)
        gammas_rad = np.array(gammas_rad)
        radii_int_pos = np.sort(dist_max/gammas_rad)
        print('Precompute(): Employing NEW discretisation method')

    radii_int_pos[0] = 0.0 #TODO: implement a proper fix
    radii_int_neg = -1*np.flip(radii_int_pos)
    radii_int_mega_high = np.array([1.0e15])
    radii_int = np.append(np.append(radii_int_pos, radii_int_mega_high), radii_int_neg)

    radii = np.zeros(parameters['num_radii'])
    count = 0
    for i, r_int in enumerate(radii_int):
        if i % 2 != 0 and i != int(len(radii_int)-1):
            radii[count] = r_int
            count +=1

    # print(f'radii = {radii}')
    
    radii_ad = np.zeros(3*parameters['num_radii']) #NOTE: intermediate radii are double here since they can have different adm_vel's
    count = 0
    for i, r_int in enumerate(radii_int):
        if i%2 != 0 and i != int(len(radii_int)-1):
            radii_ad[count] = radii_int[i-1]
            radii_ad[count+1] = radii_int[i]
            radii_ad[count+2] = radii_int[i+1]
            count = count+3

    # Setting initial vel_obs to be the limits set by v_window_max and omega_window_max
    omega_obs_init = np.zeros(len(radii_ad))
    v_obs_init = np.zeros(len(radii_ad))
    for i,r in enumerate(radii_ad):
        if r == 0.0 or r == -0.0:
            continue
        else:
            omega_obs_init[i] = np.sign(r)*min(omega_window_max, v_window_max/abs(r))
            v_obs_init[i] = min(v_window_max, omega_window_max*abs(r))
    omega_obs_init[0] = omega_window_max
    v_obs_init[0] = v_window_min
    omega_obs_init[-1] = omega_window_min
    v_obs_init[-1] = v_window_min

    omega_obs_init_mid = np.zeros(len(radii))
    v_obs_init_mid = np.zeros(len(radii))
    for i,r in enumerate(radii):
        omega_obs_init_mid[i] = np.sign(r)*min(omega_window_max, v_window_max/abs(r))
        v_obs_init_mid[i] = min(v_window_max, omega_window_max*abs(r))

    # Determining rough step indices matrix
    rough_step_size = math.floor(parameters['r_quick_check']/parameters['delta_dist'])
    if rough_step_size == 0:
        print(f'precomputations(): rough step size automatically set to 1 since delta_dist is to small')
        rough_step_size = 1
    
    # Calculating and storing precomputed values
    rough_idx_matrix = []
    point_matrix = []
    polygon_matrix = []
    dist_matrix = []
    adm_vel_matrix = []
    color_idx = 0
    for r in radii:
        # print(f'< ============= radius = {r} =================>') #DEBUG
        delta_gamma = parameters['delta_dist']/abs(r) #NOTE: dist between points deviates from the delta dist due round() in gammas
        gamma_max = min(dist_max/abs(r), 2*np.pi)
        point_num = math.floor(gamma_max/delta_gamma)
        gammas = np.linspace(0, gamma_max, point_num, endpoint=True) 
        polygon = Polygon(parameters['polygon_coords'])

        # Rough indices list
        rought_idx_list = []
        idx = 0
        while True:
            if idx >= point_num:
                break
            rought_idx_list.append(idx)
            idx += rough_step_size

        last_idx = point_num-1
        if last_idx not in rought_idx_list:
            rought_idx_list.append(last_idx)
        
        rough_idx_matrix.append(rought_idx_list)

        point_list = []
        polygon_list = []
        dist_list = []
        adm_vel_list = []
        for gamma in gammas:
            # print(f'< ============= gamma = {gamma} =================>') #DEBUG
            point = [r-r*np.cos(gamma), abs(r)*np.sin(gamma)]
            # print(f'point y = {point[1]}')
            translated_polygon = translate(polygon, point[0], point[1])
            rotated_polygon = rotate(translated_polygon, -np.sign(r)*gamma, origin=Point(point[0],point[1]), use_radians=True)
            # print(f'polygon y = {rotated_polygon.boundary.coords[0][1]}') #DEBUG

            distance = abs(r)*gamma
            # print(f'distance = {distance}') #DEBUG
            adm_vel = [math.sqrt(2*distance*abs(parameters['alpha_min'])), math.sqrt(2*distance*abs(parameters['a_min']))] # = [omega, v]
            # print(f'v_adm = {adm_vel[1]}') #DEBUG

            point_list.append(point)
            polygon_list.append(rotated_polygon)
            dist_list.append(distance)
            adm_vel_list.append(adm_vel)
            
        point_matrix.append(point_list)
        polygon_matrix.append(polygon_list)
        dist_matrix.append(dist_list)
        adm_vel_matrix.append(adm_vel_list)

        # Plotting
        if visualize: 
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan'] 
            for point, polygon in zip(point_list, polygon_list):
                ax.scatter(point[0], point[1], c=colors[color_idx % len(colors)], s=5)
                patch = plt_polygon(np.array(polygon.exterior.coords), alpha=0.1, closed=True, facecolor=colors[color_idx % len(colors)]) 
                ax.add_patch(patch)
            color_idx += 1

    # ==================== Lidar pointcloud ==================== #
    lidar_delta_psi = (2 * np.pi) / parameters['num_lidar_rays']
    lidar_angles = np.arange(0, 2 * np.pi, lidar_delta_psi)
    lidar_cosines = np.cos(lidar_angles)
    lidar_sines = np.sin(lidar_angles)
    
    # =================== Precomputed values =================== #
    precomputed = {
        "radii" : radii, 
        "radii_ad" : radii_ad, 
        "points" : point_matrix, 
        "polygons" : polygon_matrix, #TODO: change to rectangle for efficiency
        "dists" : dist_matrix, 
        "adm_vels" : adm_vel_matrix,
        "omega_obs_init" : omega_obs_init,
        "v_obs_init": v_obs_init,
        "omega_obs_init_mid": omega_obs_init_mid,
        "v_obs_init_mid": v_obs_init_mid,
        "lidar_cosines": lidar_cosines,
        "lidar_sines": lidar_sines,
        "rough_idx_matrix": rough_idx_matrix,
        "rough_step_size": rough_step_size,
        "omega_window_min": omega_window_min,
        "omega_window_max": omega_window_max,
        "v_window_min": v_window_min,
        "v_window_max": v_window_max,
        }
    precomputed['radii_ad'][0] = 1e-15 #TODO: replace with proper fix
    precomputed['radii_ad'][-1] = -1e-15
    return precomputed

def compute_velocity_obstacle(parameters, lidar_points, precomputed):
    if lidar_points.size == 0:
        return np.array([])
    
    lidar_tree = KDTree(lidar_points)
    omega_obs = precomputed['omega_obs_init'].copy()
    v_obs = precomputed['v_obs_init'].copy()

    # omega_obs_mid = precomputed['omega_obs_init_mid'].copy()
    # v_obs_mid = precomputed['v_obs_init_mid'].copy()
    omega_obs_mid = np.zeros(len(precomputed['omega_obs_init_mid']))
    v_obs_mid = np.zeros(len(precomputed['v_obs_init_mid']))

    for i in range(len(precomputed['radii'])):
        # Rough search for a collision
        rough_idx_list = precomputed['rough_idx_matrix'][i]
        for rough_idx in rough_idx_list: 
            point = precomputed['points'][i][rough_idx]
            polygon = precomputed['polygons'][i][rough_idx]

            _ , idx = lidar_tree.query(point)
            nearest_point = lidar_points[idx]
                        
            diff = point - nearest_point
            if np.dot(diff, diff) < parameters['r_quick_check'] and (polygon.contains(Point(nearest_point)) or polygon.boundary.contains(Point(nearest_point))):
                
                # Fine search for a collision
                for fine_idx in range(max(0, rough_idx-precomputed['rough_step_size']+1), len(precomputed['points'][i])-1):
                    point = precomputed['points'][i][fine_idx]
                    polygon = precomputed['polygons'][i][fine_idx]
                    
                    dist, idx = lidar_tree.query(point)
                    nearest_point = lidar_points[idx]
                    
                    diff = point - nearest_point
                    if np.dot(diff, diff) < parameters['r_quick_check'] and (polygon.contains(Point(nearest_point)) or polygon.boundary.contains(Point(nearest_point))):
                        adm_vel = precomputed['adm_vels'][i][fine_idx]

                        r_pre = precomputed['radii_ad'][3*i]
                        omega_obs[3*i] = np.sign(r_pre)*min(adm_vel[0], adm_vel[1]/abs(r_pre))
                        v_obs[3*i] = min(adm_vel[1], (adm_vel[0]*abs(r_pre)))

                        r_post = precomputed['radii_ad'][3*i+2]
                        omega_obs[3*i+2] = np.sign(r_post)*min(adm_vel[0], adm_vel[1]/abs(r_post))
                        v_obs[3*i+2] = min(adm_vel[1], (adm_vel[0]*abs(r_post)))
                        
                        r_cur = precomputed['radii_ad'][3*i+1]
                        if (v_obs[3*i] != v_obs[3*i+2]) and (omega_obs[3*i] != omega_obs[3*i+2]):
                            omega_obs[3*i+1] = np.sign(r_cur)*adm_vel[0] #NOTE: sign of r_curr correct in front facing situ?
                            v_obs[3*i+1] = adm_vel[1]
                            # mid
                            omega_obs_mid[i] = np.sign(r_cur)*adm_vel[0] #NOTE: sign of r_curr correct in front facing situ?
                            v_obs_mid[i] = adm_vel[1]
                        else:
                            v_obs[3*i+1] = min(adm_vel[1], (adm_vel[0]*abs(r_cur)))
                            omega_obs[3*i+1] = np.sign(r_cur)*min(adm_vel[0], adm_vel[1]/abs(r_cur))
                            # mid
                            v_obs_mid[i] = min(adm_vel[1], (adm_vel[0]*abs(r_cur)))
                            omega_obs_mid[i] = np.sign(r_cur)*min(adm_vel[0], adm_vel[1]/abs(r_cur))
                        break
                break

    vel_obs = np.array([omega_obs, v_obs]).T
    vel_obs_mid = np.array([omega_obs_mid, v_obs_mid]).T

    return vel_obs, vel_obs_mid

def compute_dynamic_window(parameters, cur_vel):   
    dyn_win = np.zeros(shape=(4,2))
    dyn_win[0] = np.array([
        cur_vel[0] + parameters['alpha_max']*parameters['sample_time'],
        cur_vel[1] + parameters['a_max']*parameters['sample_time']
    ])
    dyn_win[1] = np.array([
        cur_vel[0] + parameters['alpha_min']*parameters['sample_time'],
        cur_vel[1] + parameters['a_max']*parameters['sample_time']
    ])
    dyn_win[2] = np.array([
        cur_vel[0] + parameters['alpha_min']*parameters['sample_time'],
        cur_vel[1] + parameters['a_min']*parameters['sample_time']
    ])
    dyn_win[3] = np.array([
        cur_vel[0] + parameters['alpha_max']*parameters['sample_time'],
        cur_vel[1] + parameters['a_min']*parameters['sample_time']
    ])

    # (NOTE: uses the same bounds as the observation space)
    low = np.array([(parameters['omega_min'], parameters['v_min'])]*4)
    high = np.array([(parameters['omega_max'], parameters['v_max'])]*4)

    bounded_dyn_win = np.clip(a=dyn_win, a_min=low, a_max=high)
    return bounded_dyn_win

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

def compute_goal_vel_obs(parameters, local_goal_pos, cur_vel):
    if local_goal_pos[0] == 0: #NOTE: prevents singularities (only a valid assump. when goal is closeby)
        if cur_vel[0] == 0.0:
            local_goal_pos[0] = 1e-5
        else:
            local_goal_pos[0] = np.sign(cur_vel[0])*1e-5

    goal_rad = (local_goal_pos[0]**2 + local_goal_pos[1]**2)/(2*local_goal_pos[0])
    if np.sign(local_goal_pos[0]) > 0:
        psi = np.arctan(local_goal_pos[1]/local_goal_pos[0])
    else:
        psi = (np.pi/2 - abs(np.arctan(local_goal_pos[1]/local_goal_pos[0]))) + np.pi/2
    goal_gamma = np.pi - 2*psi
    goal_dist = goal_gamma*goal_rad

    omega_goal_adm = math.sqrt(2*goal_dist*abs(parameters['alpha_min']))
    v_goal_adm = math.sqrt(2*goal_dist*abs(parameters['a_min']))

    omega_goal_obs = np.sign(goal_rad)*min(omega_goal_adm, v_goal_adm/abs(goal_rad))
    v_goal_obs = min(v_goal_adm, (omega_goal_adm*abs(goal_rad)))

    return np.array([omega_goal_obs, v_goal_obs])