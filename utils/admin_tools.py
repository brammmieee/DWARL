from pathlib import Path
import yaml
from omegaconf import OmegaConf

def load_data_set_config(path_to_config):
    with open(path_to_config) as f:
        config_dict = yaml.load(f, Loader=yaml.BaseLoader)
    return OmegaConf.create(config_dict)

def load_map_name_list(path_to_map_list): 
    with open(path_to_map_list) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)

def generate_folder_structure(root_dir, path_dict):
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    for path_to in path_dict.values():
        Path(path_to).mkdir(parents=True, exist_ok=True)