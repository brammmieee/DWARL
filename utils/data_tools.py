import os
import random
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple

class PoseManager:
    def __init__(self, rng: np.random.Generator, max_combinations: int = 0):
        self.rng = rng
        self.max_combinations = max_combinations
        self.unique_combinations = set()
        self.max_combinations = 0

    def get_random_init_and_goal_pose(self, path: np.ndarray, parameters: dict) -> Tuple[np.ndarray, np.ndarray]:
        nom_dist = parameters['nominal_distance']
        
        cum_path_length = np.cumsum(np.linalg.norm(path[1:] - path[:-1], axis=1))
        init_options = np.argwhere(cum_path_length[-1] - cum_path_length > nom_dist).squeeze()
        init_index = self.rng.choice(init_options)
        goal_index = np.max(np.argwhere(cum_path_length < cum_path_length[init_index] + nom_dist))
        
        init_pose = create_pose(path[init_index], calculate_orientation(path[init_index], path[min(init_index + 1, len(path) - 1)]))
        goal_pose = create_pose(path[goal_index], calculate_orientation(path[goal_index], path[min(goal_index + 1, len(path) - 1)]))
        
        return init_pose, goal_pose

    def get_full_path_init_and_goal_pose(self, path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        init_pose = create_pose(path[0], calculate_orientation(path[0], path[min(2, len(path) - 1)]))
        goal_pose = create_pose(path[-1])
        return init_pose, goal_pose

    def get_random_distance_init_and_goal_pose(self, path: np.ndarray, parameters: dict) -> Tuple[np.ndarray, np.ndarray, int]:
        init_path_index = self.rng.integers(0, len(path) - 1)
        goal_path_index = self.rng.integers(0, len(path) - 1)
        if np.linalg.norm(path[init_path_index] - path[goal_path_index]) < parameters['min_init2goal_dist']:
            return self.get_random_distance_init_and_goal_pose(path, parameters)
        else:
            init_pose = create_pose(path[init_path_index], calculate_orientation(path[init_path_index], path[goal_path_index]))
            goal_pose = create_pose(path[goal_path_index], calculate_orientation(path[goal_path_index], path[init_path_index]))
            direction = 1 if init_path_index < goal_path_index else -1
            return init_pose, goal_pose, direction

    def sample_pose(self, path: np.ndarray, parameters: dict, mode: str) -> Tuple[np.ndarray, np.ndarray, int]:
        while len(self.unique_combinations) < self.max_combinations:
            if mode == 'random':
                init_pose, goal_pose = self.get_random_init_and_goal_pose(path, parameters)
            elif mode == 'full_path':
                init_pose, goal_pose = self.get_full_path_init_and_goal_pose(path)
            elif mode == 'random_distance':
                init_pose, goal_pose, direction = self.get_random_distance_init_and_goal_pose(path, parameters)
            else:
                raise ValueError(f"Unsupported pose sampling mode: {mode}")
            
            if (init_pose, goal_pose) not in self.unique_combinations:
                self.unique_combinations.add((init_pose, goal_pose))
                return init_pose, goal_pose, direction
        
class DataManager:
    def __init__(self, paths):
        self.paths = paths

    def load_map_raster(self, map_name: str) -> list:
        with open(self.paths.resources.maps / map_name, 'rb') as map_pgm_file:
            return read_pgm(map_pgm_file)

    def save_proto(self, map_raster: list, output_file: str):
        with open(output_file, 'w') as proto_output_file:
            convert_pgm_to_proto(map_raster, proto_output_file, output_file)

    def save_grid(self, map_raster: list, output_file: str):
        with open(output_file, 'w') as grid_output_file:
            np.save(grid_output_file, np.array(map_raster))

    def load_path(self, path_name: str) -> np.ndarray:
        with open(self.paths.resources.paths / path_name, 'r') as path_file:
            return np.load(path_file)

class DataLoader:
    def __init__(self, cfg, paths):
        self.data_manager = DataManager(paths)
        self.pose_manager = PoseManager(
            np.random.default_rng(self.cfg.data_loader.seed),
            self.cfg.data_loader.max_combinations
        )
        self.generate_data()
        self.load_data()

    def generate_data(self):  
        # load map names from list
        with open(self.paths.resources.map_name_lists / self.cfg['map']['list_name']) as f:
            map_name_list = yaml.load(f, Loader=yaml.BaseLoader)

        for map_name in map_name_list:
            # Generate map raster extracted from pgm
            map_raster = self.data_manager.load_map_raster(map_name)
            
            # Convert map raster to proto and grid and save them
            self.data_manager.save_proto(map_raster, self.paths.outputs.data_sets / 'protos')
            self.data_manager.save_grid(map_raster, self.paths.outputs.data_sets / 'grids')
        
            # iterate over all paths (for current map)
            for path_name in list(Path(self.paths.resources.paths).glob(f"{map_name}_*")):
                # Load path from text file (generated by imagej)
                path = self.data_manager.load_path(path_name)
                
                # Sample start and goal pose from path according to the pose_sampling method and cfg
                init_pose, goal_pose, direction = self.pose_manager.sample_pose(
                    path, self.cfg.data_loader, self.cfg.data_loader.pose_sampling
                )
                
                # Save the data or do other processing
                pass

    def load_data(self):
        for item in self.data:
            map_name = item['map_name']
            item['map_file'] = os.path.join(self.paths.resources.maps, f"{map_name}.wbt") if os.path.exists(os.path.join(self.paths.resources.maps, f"{map_name}.wbt")) else None
            item['grid_file'] = os.path.join(self.paths.outputs.data_sets, 'grids', f"{map_name}_grid.npy") if os.path.exists(os.path.join(self.paths.outputs.data_sets, 'grids', f"{map_name}_grid.npy")) else None
            item['path_file'] = os.path.join(self.paths.resources.paths, f"{map_name}_path.npy") if os.path.exists(os.path.join(self.paths.resources.paths, f"{map_name}_path.npy")) else None

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        item = self.data[self.index]
        self.index += 1
        return item

def create_pose(point: np.ndarray, orientation: float = 0.0) -> np.ndarray:
    """ Create a pose from a point and an orientation. """
    return np.array([point[0], point[1], 0.0, orientation])

def calculate_orientation(p1: np.ndarray, p2: np.ndarray) -> float:
    """ Calculate orientation angle from point p1 to p2. """
    align = p2 - p1
    return np.arctan2(align[1], align[0])

def read_pgm(pgmf) -> list:
    """Return a raster of integers from a PGM as a list of lists."""
    # Read header information 
    line_number = 0 
    while line_number < 2:
        line = pgmf.readline()
        if line_number == 0:  # Magic num info
            P_type = line.strip()
        if P_type != b'P2' and P_type != b'P5':
            pgmf.close()
            print('Not a valid PGM file')
            exit()
        if line_number == 1:  # Width, Height and Depth
            [width, height, depth] = (line.strip()).split()
            width = int(width)
            height = int(height)
            depth = int(depth)
        line_number += 1

    raster = []
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

def add_vertex_points(points, index, i, x1, y1, x2, y2, image_height, origin, map_res):
    """append 4 3D points to the 'points' array,
    link to these points in the 'index' array,
    together the 4 points make up a 2D vertex, front and back"""
    points.append([-origin[0] + x1*map_res, 0, (y1 - image_height)*map_res + origin[1]])
    points.append([-origin[0] + x2*map_res, 0, (y2 - image_height)*map_res + origin[1]])
    points.append([-origin[0] + x2*map_res, 1, (y2 - image_height)*map_res + origin[1]])
    points.append([-origin[0] + x1*map_res, 1, (y1 - image_height)*map_res + origin[1]])

    # link to the points in the 'points' array, end with '-1'
    index.append([i, i+1, i+2, i+3, -1])  # front side of the vertex
    index.append([i+3, i+2, i+1, i, -1])  # back side of the vertex

    return i + 4

def convert_pgm_to_proto(map_cfg, map_raster, output_file):
    origin = [-30.0 * map_cfg.resolution, 0, 0]
    origin = [-float(coord) for coord in origin]
    occupied_thresh = 255 * (1 - float(map_cfg.occupied_thresh))
    free_thresh = 255 * (1 - float(map_cfg.free_tresh))

    coords = []
    indices = []
    image_height = len(map_raster)

    i = 0
    for r, row in enumerate(map_raster[1:-1], start=1):
        for c, pixel in enumerate(row[1:-1], start=1):
            # check if pixel == wall
            if pixel < occupied_thresh:
                prev_i = i
                points = []
                index = []
                # free space above pixel?
                if map_raster[r-1][c] > free_thresh:
                    i = add_vertex_points(points, index, i, c, r, c+1, r, image_height, origin, map_cfg.resolution)
                # free space below pixel?
                if map_raster[r+1][c] > free_thresh:
                    i = add_vertex_points(points, index, i, c, r+1, c+1, r+1, image_height, origin, map_cfg.resolution)
                # free space left pixel?
                if map_raster[r][c-1] > free_thresh:
                    i = add_vertex_points(points, index, i, c, r, c, r+1, image_height, origin, map_cfg.resolution)
                # free space right pixel?
                if map_raster[r][c+1] > free_thresh:
                    i = add_vertex_points(points, index, i, c+1, r, c+1, r+1, image_height, origin, map_cfg.resolution)
                # new indexFace added?
                if i > prev_i:
                    coords.append(points)
                    indices.append(index)

    pf = open(output_file.name, 'w+')

    # Write proto-file header
    pf.write('#VRML_SIM R2019a utf8\n')
    pf.write('# license: Apache License 2.0\n')
    pf.write('# license url: http://www.apache.org/licenses/LICENSE-2.0\n')
    pf.write('\n')

    # Add appearance EXTERNPROTO
    pf.write('EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2022b/projects/robots/kuka/youbot/protos/BlackMetalAppearance.proto"')
    pf.write('\n\n')

    # Define PROTO with name 'map' and params translation and rotation
    pf.write('PROTO ' + file_name + ' [\n')
    pf.write('  field  SFVec3f     translation     0 0 0\n')
    pf.write('  field  SFRotation  rotation        1 0 0 1.5708\n')
    pf.write(']\n')
    pf.write('{\n')

    # Transform map based on params
    pf.write('  Transform {\n')
    pf.write('      translation IS translation\n')
    pf.write('      rotation    IS rotation\n')
    pf.write('      children [\n')

    # Open Shape with BlackMetalAppearance
    pf.write('          Shape {\n')
    pf.write('              appearance BlackMetalAppearance {\n')
    pf.write('              }\n')
    pf.write('              geometry IndexedFaceSet {\n')
    pf.write('                  coord Coordinate {\n')
    pf.write('                      point [\n')

    # Write all coordinates
    pf.write('                          ')
    for p in coords:
        for i in p:
            for j in i:
                pf.write(' ' + str(j))
            pf.write(',')
    pf.write('\n                      ]\n')
    pf.write('                  }\n')
    pf.write('                  coordIndex [\n')

    # Write all indices that point to the coordinates
    pf.write('                      ')
    for p in indices:
        for i in p:
            for j in i:
                pf.write(' ' + str(j))
    pf.write('\n                ]\n')
    pf.write('                  creaseAngle 0\n')
    pf.write('              }\n')
    pf.write('          }\n')
    pf.write('      ]\n')
    pf.write('  }\n')
    pf.write('}\n')

    pf.close()
    