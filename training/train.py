#!/usr/bin/python3

from environments.base_env import BaseEnv
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
import data.data_generator as dg
import data.data_set as ds
import hydra
import torch as th
import utils.wrapper_tools as wt

from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='../config', config_name='train')
def main(cfg : DictConfig):
    data_generator = dg.DataGenerator(cfg.data, cfg.paths)
    data_generator.erase_data(); data_generator.generate_data()
    data_set = ds.Dataset(cfg.data, cfg.paths)

    # print("Data configuration:")
    # print(OmegaConf.to_yaml(cfg))
    
    base_env = BaseEnv(cfg.environment, cfg.paths, data_set)
    base_env.reset()
    # import ipdb; ipdb.set_trace()
    # wrapped_env = wt.wrap_env(cfg.wrappers, base_env)
    # vec_env=make_vec_env( #NOTE: Wraps the environment in a Monitor wrapper to have additional training information
    #     env_id=wrapped_env,
    #     n_envs=cfg.envs, 
    #     vec_env_cls=SubprocVecEnv, 
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