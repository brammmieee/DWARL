import os
import random
import yaml
import numpy as np
from typing import Tuple
from pathlib import Path
import utils.data_tools as dlt

from torch.utils.data import DataLoader

import yaml

class DataLoader:
    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths
        self.data_generator = dlt.DataGenerator(cfg, paths)
        self.data = []

    def load_map_names(self):
        map_list_file = self.paths.resources.map_name_lists / self.cfg['map']['list_name']
        with open(map_list_file) as f:
            return yaml.load(f, Loader=yaml.BaseLoader)

    def prepare_data(self):
        map_names = self.load_map_names()
        self.data_generator.generate_data(map_names)

    def load_data(self):
        for item in self.data:
            map_name = item['map_name']
            # Load associated files as needed
            item['map_file'] = self.paths.resources.maps / f"{map_name}.wbt"
            item['grid_file'] = self.paths.outputs.data_sets / 'grids' / f"{map_name}_grid.npy"
            item['path_file'] = self.paths.resources.paths / f"{map_name}_path.npy"

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item