
import math
import numpy as np
import matplotlib as plt
from matplotlib.patches import Polygon as plt_polygon
from scipy.spatial import KDTree
from shapely.affinity import translate, rotate
from shapely.geometry import Point, Polygon

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
