
from pathlib import Path
import numpy as np
import yaml
import shutil

class DataGenerator:
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths

    def erase_old_data(self):
        data_sets_folder = Path(self.paths.data_sets.root)
        if data_sets_folder.exists() and data_sets_folder.is_dir():
            shutil.rmtree(data_sets_folder)
    
    def create_folder_structure(self):
        Path(self.paths.data_sets.root).mkdir(parents=True, exist_ok=True)
        for data_set_path in self.paths.data_sets.values():
            Path(data_set_path).mkdir(parents=True, exist_ok=True)
            
    def generate_data(self):
        # Generate the folder structure
        self.create_folder_structure()
        
        # Get the list of maps
        path_to_map_list = Path(self.paths.resources.map_name_lists) / f"{self.cfg.map.list}.yaml"
        with open(path_to_map_list) as f:
            map_name_list = yaml.load(f, Loader=yaml.BaseLoader)
        
        # Process each map
        for map_name in map_name_list:
            map_grid = self.pgm_to_pixel_grid(Path(self.paths.resources.maps) / f"{map_name}.pgm")
            map_box_array = self.pixel_grid_to_box_array(map_grid, self.cfg.map.resolution)
            np.save(Path(self.paths.data_sets.maps) / f"{map_name}.npy", map_box_array)
            self.process_paths_for_map(map_name, map_pixel_height=map_grid.shape[0])

    def process_paths_for_map(self, map_name, map_pixel_height):
        path_to_path_list = list(Path(self.paths.resources.paths).glob(f"{map_name}_*"))
        for path_to_path in path_to_path_list:
            path_name = Path(path_to_path).stem
            path = np.loadtxt(Path(self.paths.resources.paths) / path_to_path)
            converted_path = np.array([
                self.pixel_to_point(point, self.cfg.map.resolution, map_pixel_height) for point in path
            ])
            np.save(Path(self.paths.data_sets.paths) / f"{path_name}.npy", converted_path)
            self.generate_data_points(path_name, converted_path)

    def generate_data_points(self, path_name, converted_path):
        indices = np.arange(converted_path.shape[0])
        init_and_goal_indices = [(i, j) for i in indices for j in indices if i != j]
        data_point_indices = np.arange(len(init_and_goal_indices))

        for (init_index, goal_index), idx in zip(init_and_goal_indices, data_point_indices):
            second_point_idx = (init_index + 1) if init_index < goal_index else (init_index - 1)
            second_pos = converted_path[second_point_idx]
            init_pos = converted_path[init_index]
            goal_pos = converted_path[goal_index]
            self.create_data_points(path_name, idx, init_pos, goal_pos, second_pos)

    def create_data_points(self, path_name, idx, init_pos, goal_pos, second_pos):
        aligned_orientation = np.arctan2(second_pos[1] - init_pos[1], second_pos[0] - init_pos[0])
        reversed_orientation = np.arctan2(init_pos[1] - second_pos[1], init_pos[0] - second_pos[0])

        for orient_idx, orientation in enumerate([aligned_orientation, reversed_orientation]):
            data_point = {
                "init_pose": [float(init_pos[0]), float(init_pos[1]), float(0), float(orientation)],
                "goal_pose": [float(goal_pos[0]), float(goal_pos[1]), float(0), float(0)] # NOTE: The goal orientation is always 0
            }
            path_to_data_point = Path(self.paths.data_sets.data_points) / f"{path_name}_{idx}_{orient_idx}.yaml"
            with open(path_to_data_point, 'w') as f:
                yaml.dump(data_point, f)
      
    @staticmethod          
    def pgm_to_pixel_grid(path_to_map_pgm) -> list:
        """Return a raster of integers from a PGM as a list of lists."""
        with open(path_to_map_pgm, 'rb') as pgmf:
            # Read header information
            P_type = pgmf.readline().strip()
            if P_type not in [b'P2', b'P5']:
                raise ValueError('Not a valid PGM file')
            
            width, height, _ = map(int, pgmf.readline().strip().split())
            
            # Read raster data
            raster = [
                [ord(pgmf.read(1)) for _ in range(width)]
                for _ in range(height)
            ]
            
        return np.array(raster)
    
    @staticmethod
    def pixel_to_point(point, resolution, image_height):
        # TODO: add docstring for the axis convention
        point = np.array(point, dtype=float)
        point[:2] *= resolution  # Convert to meters
        point[1] = (resolution * image_height) - point[1]  # Invert y-axis
        return point
    
    @staticmethod
    def pixel_grid_to_box_array(pixel_grid, resolution, occupancy_threshold=0.5):
        grid_height = pixel_grid.shape[0]
        grid_width = pixel_grid.shape[1]
        
        box_array = []
        for row in range(grid_height):
            for col in range(grid_width):
                if pixel_grid[row][col] <= occupancy_threshold:
                    bl_vertex = DataGenerator.pixel_to_point([col, row], resolution, grid_height)
                    tl_vertex = DataGenerator.pixel_to_point([col, row + 1], resolution, grid_height)
                    tr_vertex = DataGenerator.pixel_to_point([col + 1, row + 1], resolution, grid_height)
                    br_vertex = DataGenerator.pixel_to_point([col + 1, row], resolution, grid_height)
                    
                    box = [bl_vertex, tl_vertex, tr_vertex, br_vertex]
                    box_array.append(box)
                    
        return np.array(box_array)
