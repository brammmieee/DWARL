#!/usr/bin/python3

# Prevent TensorFlow from spamming messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environments.base_env import BaseEnv
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from utils.data_loader import InfiniteDataLoader
from utils.webots_resource_generator import WebotsResourceGenerator
from utils.wrapper_tools import wrap_env, make_vec_env
import hydra
import subprocess
import torch as th
import utils.data_generator as dg
import utils.data_set as ds

@hydra.main(config_path='config', config_name='train', version_base='1.1')
def main(cfg: DictConfig):
    # Kill all the Webots processes that are running
    subprocess.run(["bash", str(Path(cfg.paths.scripts.killall_webots))])
            
    # Generate data
    if cfg.generate_data:
        # Generate map, path and data points in our axis convention
        data_generator = dg.DataGenerator(cfg.data_generator, cfg.paths)
        data_generator.erase_old_data()
        data_generator.generate_data()
        
        # Generate the proto and world files for the Webots environment from the generated data
        webots_resource_generator = WebotsResourceGenerator(cfg.simulation, cfg.paths)
        webots_resource_generator.erase_old_data()
        webots_resource_generator.generate_resources()
        # TODO: Update the proto files according to the configuration
    
    # Initialize the dataset and the data loader
    data_set = ds.Dataset(cfg.paths)
    data_loader = InfiniteDataLoader(data_set, cfg.envs)
    
    # Create a vectorized environment
    if cfg.envs < 2:
        raise ValueError("The number of environments (cfg.envs) must be at least 2.")
    
    vec_env=make_vec_env( #NOTE: Adds the monitor wrapper which might lead to issues with time limit wrapper! (see __init__ description)
        env_id=BaseEnv,
        n_envs=cfg.envs, 
        vec_env_cls=SubprocVecEnv, 
        wrapper_class=wrap_env,
        env_kwargs={
            'cfg': cfg.environment,
            'paths': cfg.paths,
            'sim_cfg': cfg.simulation,
            'data_loader': data_loader,
            'render_mode': None
        },
        wrapper_kwargs={
            'cfg': cfg.wrappers,
        }
    )
    
    # Initialize the model
    model=PPO(
        env=vec_env,
        policy=cfg.model.policy_type,
        tensorboard_log=cfg.paths.outputs.logs,
        policy_kwargs={
            'net_arch': OmegaConf.to_container(cfg.model.net_arch),
            'activation_fn': getattr(th.nn, cfg.model.activation_fn)
        }
    )

    # Training
    model.learn(
        total_timesteps=cfg.steps,
        callback=[
            CheckpointCallback(
                save_freq=cfg.callbacks.model_eval_freq,
                save_path=cfg.paths.outputs.models,
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=0,
            ),
            EvalCallback(
                eval_env=vec_env,
                callback_on_new_best=None,
                callback_after_eval=None,
                n_eval_episodes=cfg.callbacks.model_n_eval_episodes,
                eval_freq=cfg.callbacks.model_eval_freq,
                log_path=None,
                best_model_save_path=cfg.paths.outputs.models,
                deterministic=False,
                render=False,
                verbose=0,
                warn=True,
            )
        ],
        log_interval=cfg.log_interval,
        reset_num_timesteps=False,
        progress_bar=True
    )

if __name__ == '__main__':
    main()