from pathlib import Path
import numpy as np
import shutil
import yaml


class DataGenerator:
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths
        
        self.map_loader = MapLoader(paths)
        self.path_loader = PathLoader(paths)
        self.proto_converter = ProtoConverter(cfg.map)
        self.pose_sampler = PoseSampler(cfg.seed, cfg.pose_sampler)
        
    def erase_data(self):
        data_sets_folder = Path(self.paths.outputs.data_sets)
        if data_sets_folder.exists() and data_sets_folder.is_dir():
            shutil.rmtree(data_sets_folder)        
            
    def generate_data(self, map_names):
        # Generate folder structure for data
        proto_dir = Path(self.paths.outputs.data_sets) / "protos"
        grid_dir = Path(self.paths.outputs.data_sets) / "grids"
        data_point_dir = Path(self.paths.outputs.data_sets) / "data_points"

        # Ensure directories exist
        proto_dir.mkdir(parents=True, exist_ok=True)
        grid_dir.mkdir(parents=True, exist_ok=True)
        data_point_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate data for each map
        for map_name in map_names:
            map_raster = self.map_loader.load_map_raster(map_name)
            path_file_path_list = self.path_loader.list_paths_for_map(map_name)
            
            self.proto_converter.save_proto(
                map_raster=map_raster, 
                output_file=proto_dir / f"{map_name}.proto"
            )
            self.map_loader.save_grid(
                map_raster=map_raster, 
                output_file=grid_dir / f"{map_name}_grid.npy"
            )
            
            print(f"\n Generating data for map: {map_name}")
            print(f" Number of paths found: {len(path_file_path_list)}")
            
            # Process paths for current map
            for path_file_path in path_file_path_list:
                path = self.path_loader.load_path(path_file_path)
                path_name = Path(path_file_path).stem
                
                print(f"\t Processing path: {path_name}")
                print(f"\t Number of init goal pairs: {self.cfg.pose_sampler.max_combinations}, using mode: {self.cfg.pose_sampler.mode}")
                
                # Generate multiple unique combinations of init and goal poses per path
                self.pose_sampler.reset() # Reset unique combinations for current path
                while True:
                    init_pose, goal_pose = self.pose_sampler.sample_pose(
                        path=path,
                        mode=self.cfg.pose_sampler.mode
                    )
                    data_point = {"init_pose": init_pose.tolist(), "goal_pose": goal_pose.tolist()}
                    data_point_path = data_point_dir / f"{path_name}_{len(self.pose_sampler.unique_combinations)}.yaml"
                    with open(data_point_path, 'w') as file:
                        yaml.dump(data_point, file)
                    
                    print(f"\t\t init_pose: {init_pose}, goal_pose: {goal_pose}")
                    if len(self.pose_sampler.unique_combinations) >= self.cfg.pose_sampler.max_combinations:
                        break

class MapLoader:
    def __init__(self, paths):
        self.paths = paths

    def load_map_raster(self, map_name) -> list:
        file_path = Path(self.paths.resources.maps) / f"{map_name}.pgm"
        with open(file_path, 'rb') as map_pgm_file:
            return read_pgm(map_pgm_file)

    def save_grid(self, map_raster: list, output_file):
        np.save(output_file, np.array(map_raster))
        
        
class PathLoader:
    def __init__(self, paths):
        self.paths = paths

    def load_path(self, path_name) -> np.ndarray:
        path_file = Path(self.paths.resources.paths) / path_name
        return np.loadtxt(path_file)

    def list_paths_for_map(self, map_name) -> list:
        return list(Path(self.paths.resources.paths).glob(f"{map_name}_*"))


class ProtoConverter:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def save_proto(self, map_raster: list, output_file):
        print(f"Saving proto file to: {output_file}")
        with open(output_file, 'w') as f:
            convert_pgm_to_proto(self.cfg, map_raster, f)

    def load_proto(self, proto_name) -> np.ndarray:
        return np.load(proto_name)


class PoseSampler:
    def __init__(self, seed, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.max_combinations = cfg.max_combinations
        
    def reset(self):
        self.unique_combinations = set()
        self.sample_pose_counter = 0
        
    def sample_pose(self, path, mode):
        if mode == 'random':
            init_pose, goal_pose = self.get_random_init_and_goal_pose(path)
        elif mode == 'full_path':
            init_pose, goal_pose = self.get_full_path_init_and_goal_pose(path)
        elif mode == 'random_distance':
            init_pose, goal_pose = self.get_random_distance_init_and_goal_pose(path)

        # Check if the combination is unique
        if (tuple(init_pose), tuple(goal_pose)) not in self.unique_combinations:
            self.unique_combinations.add((tuple(init_pose), tuple(goal_pose)))
            return init_pose, goal_pose

    def get_random_init_and_goal_pose(self, path):
        nom_dist = self.cfg.nominal_distance
        
        cum_path_length = np.cumsum(np.linalg.norm(path[1:] - path[:-1], axis=1))
        init_options = np.argwhere(cum_path_length[-1] - cum_path_length > nom_dist).squeeze()
        init_index = self.rng.choice(init_options)
        goal_index = np.max(np.argwhere(cum_path_length < cum_path_length[init_index] + nom_dist))
        
        init_pose = self.create_pose(path[init_index], self.calculate_orientation(path[init_index], path[min(init_index + 1, len(path) - 1)]))
        goal_pose = self.create_pose(path[goal_index], self.calculate_orientation(path[goal_index], path[min(goal_index + 1, len(path) - 1)]))
        
        return init_pose, goal_pose

    def get_full_path_init_and_goal_pose(self, path):
        init_pose = self.create_pose(path[0], self.calculate_orientation(path[0], path[min(2, len(path) - 1)]))
        goal_pose = self.create_pose(path[-1])
        
        # Reverse the order of the poses
        if self.sample_pose_counter == 1:
            return goal_pose, init_pose
        if self.sample_pose_counter > 1:
            raise ValueError("Full path mode only supports 2 unique combinations")
        self.sample_pose_counter += 1
        
        return init_pose, goal_pose

    def get_random_distance_init_and_goal_pose(self, path):
        init_path_index = self.rng.integers(0, len(path) - 1)
        goal_path_index = self.rng.integers(0, len(path) - 1)
        if np.linalg.norm(path[init_path_index] - path[goal_path_index]) < self.cfg.min_init2goal_dist:
            return self.get_random_distance_init_and_goal_pose(path)
        else:
            init_pose = self.create_pose(path[init_path_index], self.calculate_orientation(path[init_path_index], path[goal_path_index]))
            goal_pose = self.create_pose(path[goal_path_index], self.calculate_orientation(path[goal_path_index], path[init_path_index]))
            return init_pose, goal_pose
            
    @staticmethod        
    def create_pose(point, orientation: float = 0.0) -> np.ndarray:
        """ Create a pose from a point and an orientation. """
        return np.array([point[0], point[1], 0.0, orientation])

    @staticmethod
    def calculate_orientation(p1, p2) -> float:
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
    pf.write('PROTO ' + "DEBBUG" + ' [\n') # NOTE !!!!!!!!! # NOTE !! 
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
    