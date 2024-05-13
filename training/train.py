#!/usr/bin/python3

from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

import utils.admin_tools as at
from environments.base_env import BaseEnv
from environments.wrappers.sparse_lidar_observation_wrapper import SparseLidarObservationWrapper as SLObsWrapper
from environments.wrappers.command_velocity_action_wrapper import CommandVelocityActionWrapper as CVActWrapper
from environments.wrappers.velocity_obstacle_observation_wrapper import VelocityObstacleObservationWrapper as VObsWrapper
from environments.wrappers.dynamic_window_action_wrapper import DynamicWindowActionWrapper as DWActWrapper
from environments.wrappers.parameterized_reward_wrapper import ParameterizedRewardWrapper as PRewWrapper

def chain_wrappers(env, wrapper_classes):
    for wrapper_class in wrapper_classes:
        env = wrapper_class(env)
        
    return env

def main():
    # Environment settings
    n_envs = 2
    env_render_mode = None
    env_wb_open = True
    env_wb_mode = 'training'
    env_proto_config = 'sparse_lidar_proto_config.json'
    env_class = BaseEnv
    wrapper_class = [SLObsWrapper, CVActWrapper, PRewWrapper]
    
    # Model settings
    test_nr_today = 0
    comment = 'test'
    model_name = at.get_file_name_with_date(test_nr_today, comment)
    policy_type = "MlpPolicy"
    
    # Train and callback settings
    total_training_steps = 1e3
    model_save_freq = 5000 # [steps]
    model_eval_freq = 10000 # [steps]
    model_n_eval_episodes = 25 
    
    # Creating vectorized environment
    vec_env = make_vec_env(
        env_id=env_class,
        wrapper_class=lambda env: chain_wrappers(env, wrapper_class),
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            'render_mode': env_render_mode, 
            'wb_open': env_wb_open, 
            'wb_mode': env_wb_mode,
            'proto_config': env_proto_config,
        }
    )
    
    # Creating PPO model with callbacks
    model = PPO(
        policy=policy_type,
        env=vec_env,
        tensorboard_log = "./logs/" + model_name,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq = model_save_freq,
        save_path = "./models/" + model_name,
        name_prefix = model_name,
        save_replay_buffer = False,
        save_vecnormalize = False,
        verbose = 0,
    )

    eval_callback = EvalCallback(
        eval_env = vec_env,
        callback_on_new_best = None,
        callback_after_eval = None,
        n_eval_episodes = model_n_eval_episodes,
        eval_freq = model_eval_freq,
        log_path = None,
        best_model_save_path = "./models",
        deterministic = False,
        render = False,
        verbose = 0,
        warn = True,
    )

    # Train model
    model.learn(
        total_timesteps=total_training_steps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        tb_log_name=model_name,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
if __name__=='__main__':
    main()