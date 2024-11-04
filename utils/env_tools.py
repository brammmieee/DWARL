from controller import Supervisor
from pathlib import Path
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

WEBOTS_ROBOT_Z_POS = 0.05

def compute_collision_detection_tree(gridmap, resolution):
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

def compute_map_bound_polygon(res, width, height, padding):
    map_bounds = [
        [0.0, 0.0], 
        [(width * res - res), 0.0], 
        [(width * res - res), (height * res - res)], 
        [0.0, (height * res - res)]
    ]

    map_bound_polygon = Polygon(map_bounds)
    buffered_map_bound_polygon = map_bound_polygon.buffer(padding)
    
    return Polygon(buffered_map_bound_polygon)

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

    # Compensate for different axis convention compared to Webots
    rot_z = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) 
    cur_orient_matrix = np.dot(rot_z, cur_orient_matrix) # 90-degree rotation around the z-axis
    
    # Calculate the goal position in the local frame of reference
    local_goal_pos = np.dot(cur_orient_matrix.T, translation)
    return local_goal_pos

def compute_new_pose(dt, cur_pos, cur_orient_matrix, cur_vel):
    '''
    Kinematic compution of the new pose based on the current pose and velocity.
    '''
    # Computing the orientation
    psi = (np.arctan2(cur_orient_matrix[3], cur_orient_matrix[0])) % (2*np.pi)
    local_rotation = cur_vel[0]*dt
    global_rotation = -local_rotation #NOTE: minus to account for our conventions
    orientation = psi + global_rotation 
    
    # Computing the position
    distance = cur_vel[1]*dt
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
        WEBOTS_ROBOT_Z_POS
        ])

    return position, orientation

##### NOTE!!! are we sure the get velocity call actually gives us a velocity? #####
def get_cmd_vel(robot_node):
    webots_vel = robot_node.getVelocity()
    ang_vel = (-1)*webots_vel[-1] #NOTE: times (-1) because clockwise rotation is taken as the positve direction
    lin_vel = np.sqrt(webots_vel[0]**2 + webots_vel[1]**2) # in plane global velocities (x and y) to forward vel
    return np.array([ang_vel, lin_vel])

def apply_kinematic_constraints(dt, cfg, cur_vel, target_vel):

    domega = target_vel[0] - cur_vel[0]
    domega_clipped = np.clip(domega, cfg.alpha_min*dt, cfg.alpha_max*dt) 
    omega_clipped = np.clip((cur_vel[0] + domega_clipped), cfg.omega_min, cfg.omega_max)

    dv = target_vel[1] - cur_vel[1]
    dv_clipped = np.clip(dv, cfg.a_min*dt, cfg.a_max*dt)
    v = np.clip((cur_vel[1] + dv_clipped), cfg.v_min, cfg.v_max)

    return np.array([omega_clipped, v])

def lidar_precomputation(num_lidar_rays):
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