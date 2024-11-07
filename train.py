#!/usr/bin/python3

# Prevent TensorFlow from spamming messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environments.base_env import BaseEnv
from environments.webots_env import WebotsEnv
from omegaconf import DictConfig
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from utils.data_loader import InfiniteDataLoader
from utils.webots_resource_generator import WebotsResourceGenerator
from utils.wrapper_tools import wrap_env
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
    infinite_loader = InfiniteDataLoader(data_set, cfg.envs)
    
    # Create a vectorized environment
    def make_env(env_idx, env_kwargs, wrapper_kwargs):
        """ Based on stable baselines3's make_vec_env """
        sim_env = WebotsEnv(cfg.simulation, cfg.paths)
        return lambda: wrap_env(
            Monitor(BaseEnv(**env_kwargs, env_idx=env_idx, sim_env=sim_env)),
            **wrapper_kwargs
        )
    env_kwargs = {
        'cfg': cfg.environment,
        'paths': cfg.paths,
        'sim_env': WebotsEnv(cfg.simulation, cfg.paths),
        'data_loader': infinite_loader,
    }
    wrapper_kwargs = {
        'cfg': cfg.wrappers,
    }
    vec_env = SubprocVecEnv([
        make_env(env_idx, env_kwargs, wrapper_kwargs)
        for env_idx in range(cfg.envs-1)
    ])
    
    # Initialize the model
    model=PPO(
        env=vec_env,
        policy=cfg.model.policy_type,
        tensorboard_log=cfg.paths.outputs.logs,
        policy_kwargs={
            'net_arch': cfg.model.net_arch,
            'activation_fn': getattr(th.nn, cfg.model.activation_fn)
        }
    )

    # Training
    model.learn(
        total_timesteps=cfg.steps,
        callback=[
            CheckpointCallback(
                save_freq=cfg.callbacks.model_eval_freq,
                save_path=cfg.outputs.models,
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
                best_model_save_path=cfg.ouputs.models,
                deterministic=False,
                render=False,
                verbose=0,
                warn=True,
            )
        ],
        log_interval=cfg.callbacks.log_interval,
        reset_num_timesteps=False,
        progress_bar=True
    )

if __name__ == '__main__':
    main()