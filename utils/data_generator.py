
from pathlib import Path
import numpy as np
import shutil
import yaml
from omegaconf import DictConfig, OmegaConf
from utils.admin_tools import load_map_name_list, generate_folder_structure, convert_point_from_image_base, load_data_set_config
    
def convert_path_from_image_to_base(path, resolution, image_height):
    converted_path = []
    for point in path:
        point = convert_point_from_image_base(point, resolution, image_height)
        converted_path.append(point)
    return np.array(converted_path)

def convert_pgm_to_grid(pgmf) -> list:
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
    return np.array(raster)


class DataGenerator:
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths
    
    def prepare_and_generate_data(self):
        if not self.check_for_data():
            print("Data Generator - Data does not exist with the configuration provided. Erasing old data and generating new data.")
            self.erase_data()
            print("Data Generator - Generating folder structure")
            generate_folder_structure(self.paths.data_sets.root, self.paths.data_sets)
            self.save_config()
            self.generate_data()
        else:
            print("Data Generator - Data already exists with the configuration provided. "
                  "Please make sure that in the meantime you have not changed "
                  "the map list nor the maps and/or path mentioned in the map list.")
    
    def check_for_data(self):
        if not Path(self.paths.data_sets.config).exists():
            return False
        
        path_to_config = Path(self.paths.data_sets.config) / "config.yaml"
        old_cfg = load_data_set_config(path_to_config)
        
        if old_cfg != OmegaConf.to_yaml(self.cfg):
            return False
        
        return True
        
    def save_config(self):
        path_to_config = Path(self.paths.data_sets.config) / "config.yaml"
        with open(path_to_config, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(self.cfg), f)
        
    def erase_data(self):
        print("Erasing data")
        data_sets_folder = Path(self.paths.data_sets.root)
        if data_sets_folder.exists() and data_sets_folder.is_dir():
            shutil.rmtree(data_sets_folder)
        
    def load_grid_from_pgm(self, map_name):
        path_to_map_pgm = Path(self.paths.resources.maps) / f"{map_name}.pgm"
        with open(path_to_map_pgm, 'rb') as map_pgm_file:
            return convert_pgm_to_grid(map_pgm_file)
    
    def save_grid_to_npy(self, map_name, map_grid):
        path_to_grid = Path(self.paths.data_sets.grids) / f"{map_name}_grid.npy"
        np.save(path_to_grid,map_grid)
        
    def load_path_from_txt(self, path_name):
        path_file = Path(self.paths.resources.paths) / path_name
        return np.loadtxt(path_file)
    
    def save_path_to_npy(self, path_name, path):
        path_file = Path(self.paths.data_sets.paths) / f"{path_name}.npy"
        np.save(path_file, path)
        
    def save_data_point(self, path_name, init_pose, goal_pose, data_point_idx):
        data_point = {"init_pose": init_pose.tolist(), "goal_pose": goal_pose.tolist()}
        path_to_data_point = Path(self.paths.data_sets.data_points) / f"{path_name}_{data_point_idx}.yaml"
        with open(path_to_data_point, 'w') as file:
            yaml.dump(data_point, file)
            
    def generate_data(self):
        print("Data Generator - Generating data")
        # Generate data for each map
        path_to_map_list = Path(self.paths.resources.map_name_lists) / f"{self.cfg.map.list}.yaml"  
        for map_name in load_map_name_list(path_to_map_list):
            # Load map from pgm and save it as a numpy array 
            map_grid = self.load_grid_from_pgm(map_name)
            self.save_grid_to_npy(map_name, map_grid)
            # image_height = map_grid.shape[0] # [px]
            
            # Loop over all paths belonging to the map
            path_to_path_list = list(Path(self.paths.resources.paths).glob(f"{map_name}_*"))
            for path_to_path in path_to_path_list:
                path_name = Path(path_to_path).stem
                
                # Load path from txt, convert it and save it as a numpy array
                path = self.load_path_from_txt(path_to_path)
                converted_path = convert_path_from_image_to_base(path, self.cfg.map.resolution, map_grid.shape[0])
                self.save_path_to_npy(path_name, converted_path)
                
                # Loop over all unique combinations of poses
                indices = np.arange(path.shape[0])
                init_and_goal_indices = [(i, j) for i in indices for j in indices if i != j]
                data_point_indices = np.arange(len(init_and_goal_indices))

                for (init_index, goal_index), idx in zip(init_and_goal_indices, data_point_indices):
                    sign = 1 if init_index < goal_index else -1
                    second_point_idx = init_index + sign

                    second_pos = converted_path[second_point_idx]
                    init_pos = converted_path[init_index]
                    goal_pos = converted_path[goal_index]
                    
                    # Create pose that aligns the robot with the first path segment
                    aligned_orientation = np.arctan2(second_pos[1] - init_pos[1], second_pos[0] - init_pos[0])
                    init_pose = np.array([init_pos[0], init_pos[1], 0, aligned_orientation])
                    goal_pose = np.array([goal_pos[0], goal_pos[1], 0, 0]) # NOTE: we do nothing yet with the goal orientation
                    self.save_data_point(path_name, init_pose, goal_pose, data_point_idx=idx)
                    
                    # Create pose that aligns the robot with the first path segment but in the opposite direction
                    reversed_orientation = np.arctan2(init_pos[1] - second_pos[1], init_pos[0] - second_pos[0])
                    init_pose = np.array([init_pos[0], init_pos[1], 0, reversed_orientation])
                    goal_pose = np.array([goal_pos[0], goal_pos[1], 0, 0]) # NOTE: we do nothing yet with the goal orientation
                    self.save_data_point(path_name, init_pose, goal_pose, data_point_idx=idx)
                    
        print("Data Generator - Data generation finished")