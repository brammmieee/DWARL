import torch
import numpy as np
from pathlib import Path
import yaml

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths
        
        data_point_dir = Path(paths.data_sets.data_points)
        self.data_points = sorted(data_point_dir.glob("*.yaml"))

    def __len__(self):
        return len(self.data_points)

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
        data_point_path = self.data_points[idx]
        path_name = data_point_path.stem
        
        # Use the parsing method to extract components
        dataset_name, map_idx, path_idx, data_point_idx = self._parse_path_name(path_name)

        # Load grid
        grid_file_path = Path(self.paths.data_sets.grids) / f"{dataset_name}_{map_idx}_grid.npy"
        grid = np.load(grid_file_path)

        # Load start and goal poses from YAML
        with open(data_point_path, 'r') as f:
            data_point = yaml.safe_load(f)
            init_pose = np.array(data_point["init_pose"])
            goal_pose = np.array(data_point["goal_pose"])

        return {
            "grid": grid,
            "proto_name": f"{dataset_name}_{map_idx}",
            "init_pose": init_pose,
            "goal_pose": goal_pose,
        }
