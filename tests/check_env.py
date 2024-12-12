# Prevent TensorFlow from spamming messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environments.base_env import BaseEnv
from omegaconf import DictConfig
from utils.data_loader import InfiniteDataLoader
from utils.webots_resource_generator import WebotsResourceGenerator
import hydra
import utils.data_generator as dg
import utils.data_set as ds
import numpy as np
from stable_baselines3.common.env_checker import check_env
import time
from utils.wrapper_tools import wrap_env

@hydra.main(config_path='../config', config_name='check_env', version_base='1.1')
def main(cfg: DictConfig):
    # Generate map, path and data points in our axis convention
    data_generator = dg.DataGenerator(cfg.data_generator, cfg.paths, cfg.seed)
    data_generator.erase_old_data()
    data_generator.generate_data()

    # Generate the proto and world files for the Webots environment from the generated data
    webots_resource_generator = WebotsResourceGenerator(cfg.setup.simulation, cfg.paths)
    webots_resource_generator.erase_old_data()
    webots_resource_generator.generate_resources()
                
    # Initialize the dataset and the data loader
    data_set = ds.Dataset(cfg.paths)
    data_loader = InfiniteDataLoader(data_set, num_envs=1, seed=cfg.seed)
    # Creating wrapped environment
    env = wrap_env(
        BaseEnv(
            cfg=cfg.setup.environment,
            paths=cfg.paths,
            sim_cfg=cfg.setup.simulation,
            data_loader=data_loader,
            env_idx=0,
            render=True,
        ), 
        cfg.setup.wrappers
    )
    
    # Check the environment
    # check_env(env)

    # Run the environment for a few steps to visually check it
    env.reset()
    for _ in range(100000000):
        # import ipdb; ipdb.set_trace()
        env.step(np.array([0.0, 0.0]))
        time.sleep(0.05)

if __name__ == "__main__":
    main()