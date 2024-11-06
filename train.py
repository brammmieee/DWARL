#!/usr/bin/python3

from environments.base_env import BaseEnv
from environments.webots_env import WebotsEnv
from functools import partial
from omegaconf import DictConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from utils.data_loader import InfiniteDataLoader
from utils.webots_resource_generator import WebotsResourceGenerator
from utils.wrapper_tools import wrap_env
import hydra
import torch as th
import utils.data_generator as dg
import utils.data_set as ds

@hydra.main(config_path='config', config_name='train')
def main(cfg : DictConfig):
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
    
    data_set = ds.Dataset(cfg.paths)
    infinite_loader = InfiniteDataLoader(data_set, cfg.envs)
    
    env=BaseEnv(
        cfg=cfg.environment,
        paths=cfg.paths,
        sim_env=WebotsEnv(cfg.simulation, cfg.paths),
        data_loader=infinite_loader,
        env_idx=0
    )
    
    # env = wrap_env(env, cfg.wrappers)
    
    import numpy as np
    # for i in range(300):
    env.reset()
    while True:
        obs, rew, done, _, _  = env.step(np.array([0.4, 0.3]))
        if done:
            env.reset()


    # vec_env=make_vec_env( #NOTE: Adds the monitor wrapper which might lead to issues with time limit wrapper! (see __init__ description)
    #     env_id=BaseEnv,
    #     n_envs=cfg.envs, 
    #     vec_env_cls=SubprocVecEnv, 
    #     wrapper_class=env_wrapper,
    #     env_kwargs={
    #         'cfg': cfg.environment,
    #         'paths': cfg.paths,
    #         'sim_env': WebotsEnv(cfg.simulation, cfg.paths),
    #         'data_loader': infinite_loader,
    #         'env_idx': 0 # TODO: Change this to a list of indices
    #     },
    #     wrapper_kwargs={
    #         'cfg': cfg.wrappers,
    #     }
    # )

    # model=PPO(
    #     env=vec_env,
    #     policy=cfg.model.policy_type,
    #     tensorboard_log=cfg.paths.outputs.logs,
    #     policy_kwargs={
    #         'net_arch': cfg.model.net_arch,
    #         'activation_fn': getattr(th.nn, cfg.model.activation_fn)
    #     }
    # )

    # # Training
    # model.learn(
    #     total_timesteps=cfg.steps,
    #     callback=[
    #         CheckpointCallback(
    #             save_freq=cfg.callbacks.model_eval_freq,
    #             save_path=cfg.outputs.models,
    #             save_replay_buffer=False,
    #             save_vecnormalize=False,
    #             verbose=0,
    #         ),
    #         EvalCallback(
    #             eval_env=vec_env,
    #             callback_on_new_best=None,
    #             callback_after_eval=None,
    #             n_eval_episodes=cfg.callbacks.model_n_eval_episodes,
    #             eval_freq=cfg.callbacks.model_eval_freq,
    #             log_path=None,
    #             best_model_save_path=cfg.ouputs.models,
    #             deterministic=False,
    #             render=False,
    #             verbose=0,
    #             warn=True,
    #         )
    #     ],
    #     log_interval=cfg.callbacks.log_interval,
    #     reset_num_timesteps=False,
    #     progress_bar=True
    # )

if __name__ == '__main__':
    main()