#!/usr/bin/python3

import argparse
import os
import yaml

from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

import utils.admin_tools as at
from environments.base_env import BaseEnv
from environments.wrappers.sparse_lidar_observation_wrapper import SparseLidarObservationWrapper
from environments.wrappers.command_velocity_action_wrapper import CommandVelocityActionWrapper
from environments.wrappers.velocity_obstacle_observation_wrapper import VelocityObstacleObservationWrapper
from environments.wrappers.dynamic_window_action_wrapper import DynamicWindowActionWrapper
from environments.wrappers.parameterized_reward_wrapper import ParameterizedRewardWrapper

def chain_wrappers(env, wrapper_classes):
    for wrapper_class in wrapper_classes:
        env = wrapper_class(env)
        
    return env

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    
    # Environment settings
    parser.add_argument('--n_envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--env_proto_config', type=str, default='sparse_lidar_proto_config.json', help='Proto config file to adjust proto settings')
    parser.add_argument('--wrapper_classes', nargs='+', default=['SparseLidarObservationWrapper', 'CommandVelocityActionWrapper', 'ParameterizedRewardWrapper'], help='List of wrapper classes around BaseEnv that are applied chronologically')
    
    # Model settings
    parser.add_argument('--comment', type=str, default='test', help='Comment that hints the setup used for training')
    parser.add_argument('--policy_type', type=str, default='MlpPolicy', help='Policy type used for configuring the model')
    
    # Train and callback settings
    parser.add_argument('--total_training_steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--model_save_freq', type=int, default=5000, help='Model save frequency')
    parser.add_argument('--model_eval_freq', type=int, default=10000, help='Model evaluation frequency')
    parser.add_argument('--model_n_eval_episodes', type=int, default=25, help='Number of evaluation episodes')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    
    args = parser.parse_args()
    return args

def save_args_to_yaml(args, dir):
    # Group arguments under the specified categories
    config = {
        'comment': args.comment,
        'environment': {
            'n_envs': args.n_envs,
            'env_proto_config': args.env_proto_config,
            'wrapper_classes': args.wrapper_classes,
        },
        'model': {
            'policy_type': args.policy_type,
        },
        'train_and_callback': {
            'total_training_steps': args.total_training_steps,
            'model_save_freq': args.model_save_freq,
            'model_eval_freq': args.model_eval_freq,
            'model_n_eval_episodes': args.model_n_eval_episodes,
            'log_interval': args.log_interval,
        }
    }

    # Create the directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    config_file = os.path.join(dir, "args.yaml")

    # Save the grouped args to a yaml file
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def train(args, model_dir, log_dir):
    # Environment settings
    n_envs = args.n_envs
    env_proto_config = args.env_proto_config
    wrapper_class = [globals()[wrapper] for wrapper in args.wrapper_classes]
    
    # Model settings
    policy_type = args.policy_type
    
    # Train and callback settings
    total_training_steps = args.total_training_steps
    model_save_freq = args.model_save_freq
    model_eval_freq = args.model_eval_freq
    model_n_eval_episodes = args.model_n_eval_episodes
    log_interval = args.log_interval

    # Creating vectorized environment
    vec_env = make_vec_env(
        env_id=BaseEnv,
        wrapper_class=lambda env: chain_wrappers(env, wrapper_class),
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            'proto_config': env_proto_config,
        }
    )
    
    # Creating PPO model with callbacks
    model = PPO(
        policy=policy_type,
        env=vec_env,
        tensorboard_log = log_dir,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq = model_save_freq,
        save_path = model_dir,
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
        best_model_save_path = model_dir,
        deterministic = False,
        render = False,
        verbose = 0,
        warn = True,
    )

    # Train model
    model.learn(
        total_timesteps=total_training_steps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=log_interval,
        tb_log_name=log_dir,
        reset_num_timesteps=False,
        progress_bar=True
    )

def main():
    # Parse training arguments
    args = parse_args()

    # Directories for bookkeeping
    date_time = at.get_date()
    config_dir = f'./archive/configs/{date_time}'
    model_dir = f'./archive/models/{date_time}'
    log_dir = f'./archive/logs/{date_time}'

    # Save args to yaml file
    save_args_to_yaml(args, config_dir)

    # Train using parameters parsed
    train(args, model_dir, log_dir)
    
if __name__=='__main__':
    main()