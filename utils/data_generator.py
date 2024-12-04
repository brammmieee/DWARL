
from pathlib import Path
import numpy as np
import yaml
import shutil

class DataGenerator:
    """ The data generator is responsibe for generating the maps, paths and data points in our axis convention
    from the resource files, i.e. the map.pgm and the path.txt files."""
    def __init__(self, cfg, paths, seed):
        self.cfg = cfg
        self.paths = paths
        self.seed = seed

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
            self.generate_data_points(path_name, converted_path, self.cfg.path_sampler, self.seed)

    def generate_data_points(self, path_name, converted_path, cfg, seed):
        if cfg.name == 'all':
            self.generate_all_combinations(path_name, converted_path, cfg)
        elif cfg.name == 'start2finish':
            self.generate_start2finish(path_name, converted_path, cfg)
        elif cfg.name == 'semi_random':
            self.generate_semi_random(path_name, converted_path, cfg, seed)
        else:
            raise ValueError(f"Unknown path sampling strategy: {cfg.name}")
            
    def generate_all_combinations(self, path_name, converted_path, cfg):
        indices = np.arange(converted_path.shape[0])
        init_and_goal_indices = [(i, j) for i in indices for j in indices if i != j]
        data_point_indices = np.arange(len(init_and_goal_indices))

        for (init_index, goal_index), idx in zip(init_and_goal_indices, data_point_indices):
            second_point_idx = (init_index + 1) if init_index < goal_index else (init_index - 1)
            second_pos = converted_path[second_point_idx]
            init_pos = converted_path[init_index]
            goal_pos = converted_path[goal_index]
            self.create_data_points(path_name, idx, init_pos, goal_pos, second_pos, cfg.reverse_align)

    def generate_start2finish(self, path_name, converted_path, cfg):
        init_pos = converted_path[0]
        goal_pos = converted_path[-1]
        second_pos = converted_path[1]
        
        self.create_data_points(path_name, 0, init_pos, goal_pos, second_pos, cfg.reverse_align)
        
        if cfg.reverse_path:
            second_pos_reverse = converted_path[-2]
            self.create_data_points(path_name, 1, goal_pos, init_pos, second_pos_reverse, cfg.reverse_align)

    def generate_semi_random(self, path_name, converted_path, cfg, seed):
        # Initialize random number generator
        rng = np.random.default_rng(seed)
        
        path_length = converted_path.shape[0]
        
        # Calculate actual path distances between consecutive points
        path_segments = np.diff(converted_path, axis=0)
        segment_distances = np.sqrt(np.sum(path_segments**2, axis=1))
        cumulative_distances = np.concatenate(([0], np.cumsum(segment_distances)))
        total_path_length = cumulative_distances[-1]
        
        # Set maximum distance if not specified
        if cfg.max_dist is None:
            max_dist = total_path_length
        else:
            max_dist = min(cfg.max_dist, total_path_length)
                
        # Set mean and std if not specified
        if cfg.dist_mean is None:
            dist_mean = (cfg.min_dist + max_dist) / 2
        if cfg.dist_std is None:
            dist_std = (max_dist - cfg.min_dist) / 4

        sampled_pairs = set()
        attempts = 0
        data_point_idx = 0  # Initialize the index counter here
        
        while len(sampled_pairs) < cfg.nr_points and attempts < cfg.nr_attempts:
            # Sample desired distance in meters
            desired_distance = np.clip(rng.normal(dist_mean, dist_std), cfg.min_dist, max_dist)
            
            # Sample start position along path (in meters)
            max_start = total_path_length - desired_distance
            if max_start <= 0:
                attempts += 1
                continue
                    
            start_dist = rng.uniform(0, max_start)
            end_dist = start_dist + desired_distance
            
            # Convert distances to indices
            init_index = np.searchsorted(cumulative_distances, start_dist)
            goal_index = np.searchsorted(cumulative_distances, end_dist)
            
            # Ensure we have valid indices
            if init_index >= path_length - 1 or goal_index >= path_length:
                attempts += 1
                continue

            pair = (init_index, goal_index)
            if pair not in sampled_pairs:
                sampled_pairs.add(pair)
                second_point_idx = init_index + 1
                
                init_pos = converted_path[init_index]
                goal_pos = converted_path[goal_index]
                second_pos = converted_path[second_point_idx]
                
                self.create_data_points(path_name, data_point_idx, init_pos, goal_pos, second_pos, cfg.reverse_align)
                
                if cfg.reverse_path:
                    second_pos_reverse = converted_path[goal_index - 1]
                    self.create_data_points(path_name, data_point_idx + 1, goal_pos, init_pos, second_pos_reverse, cfg.reverse_align)
                    data_point_idx += 2
                else:
                    data_point_idx += 1
                
            attempts += 1

    def create_data_points(self, path_name, idx, init_pos, goal_pos, second_pos, reverse_align):
        # Calculate forward orientation
        dy = second_pos[1] - init_pos[1]
        dx = second_pos[0] - init_pos[0]
        aligned_orientation = np.arctan2(dy, dx)
        
        orientations = [aligned_orientation]
        if reverse_align:
            reversed_orientation = aligned_orientation + np.pi
            # Normalize to [-pi, pi] range
            if reversed_orientation > np.pi:
                reversed_orientation -= 2 * np.pi
            orientations.append(reversed_orientation)

        for orient_idx, orientation in enumerate(orientations):
            data_point = {
                "init_pose": [float(init_pos[0]), float(init_pos[1]), float(0), float(orientation)],
                "goal_pose": [float(goal_pos[0]), float(goal_pos[1]), float(0), float(0)] # NOTE: goal orientation is not used
            }
            path_to_data_point = Path(self.paths.data_sets.data_points) / f"{path_name}_{idx}_{orient_idx}.yaml"
            with open(path_to_data_point, 'w') as f:
                yaml.dump(data_point, f)
        
    @staticmethod
    def pgm_to_pixel_grid(path_to_map_pgm) -> np.ndarray:
        """Return a raster of integers from a PGM as a numpy array."""
        with open(path_to_map_pgm, 'rb') as pgmf:
            # Read header information
            P_type = pgmf.readline().strip()
            if P_type not in [b'P2', b'P5']:
                raise ValueError('Not a valid PGM file')
            
            # Skip comments
            line = pgmf.readline()
            while line.startswith(b'#'):
                line = pgmf.readline()
            
            # Read width, height, and max value
            dimensions = line.split()
            if len(dimensions) == 2:
                width, height = map(int, dimensions)
                max_val = int(pgmf.readline().strip())
            elif len(dimensions) == 3:
                width, height, max_val = map(int, dimensions)
            else:
                raise ValueError('Invalid PGM header format')
            
            # Read raster data
            if P_type == b'P5':
                raster = np.frombuffer(pgmf.read(), dtype=np.uint8).reshape(height, width)
            else:  # P2
                raster = np.array([list(map(int, pgmf.readline().split())) for _ in range(height)])
            
        return raster

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