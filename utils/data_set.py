import torch
import numpy as np
from pathlib import Path
import yaml

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        path_to_data_points = Path(paths.data_sets.data_points)
        self.path_to_datapoint_list = sorted(path_to_data_points.glob("*.yaml"))

    def __len__(self):
        return len(self.path_to_datapoint_list)

    def _parse_path_name(self, path_name):
        # Split from the end to handle dataset names with underscores
        parts = path_name.rsplit('_', 3)
        
        if len(parts) < 4:
            raise ValueError(f"Unexpected format in path name: {path_name}")
        
        dataset_name = '_'.join(parts[:-3])
        map_idx = parts[-3]
        path_idx = parts[-2]
        data_point_idx = parts[-1]
        
        return dataset_name, map_idx, path_idx, data_point_idx

    def __getitem__(self, idx):
        data_point_path = self.path_to_datapoint_list[idx]
        data_point_name = data_point_path.stem
        
        # Use the parsing method to extract components
        dataset_name, map_idx, path_idx, data_point_idx = self._parse_path_name(data_point_name)
        proto_name = f"{dataset_name}_{map_idx}"

        # Load grid
        grid_file_path = Path(self.paths.data_sets.grids) / f"{dataset_name}_{map_idx}_grid.npy"
        grid = np.load(grid_file_path)
        
        # Load path
        path_file_path = Path(self.paths.data_sets.paths) / f"{dataset_name}_{map_idx}_{path_idx}.npy"
        path = np.load(path_file_path)
        
        # Load start and goal poses
        with open(data_point_path, 'r') as f:
            data_point = yaml.safe_load(f)
            init_pose = np.array(data_point["init_pose"])
            goal_pose = np.array(data_point["goal_pose"])

        return {
            "proto_name": proto_name,
            "grid": grid,
            "path": path,
            "init_pose": init_pose,
            "goal_pose": goal_pose,
        }
