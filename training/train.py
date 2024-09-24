#!/usr/bin/python3

import argparse
import os
import sys
import yaml
import torch as th
from gymnasium.wrappers import TimeLimit
# import tensorrt

from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

import utils.admin_tools as at
import utils.base_tools as bt
from environments.base_env import BaseEnv
from environments.wrappers.sparse_lidar_observation_wrapper import SparseLidarObservationWrapper
from environments.wrappers.command_velocity_action_wrapper import CommandVelocityActionWrapper
from environments.wrappers.velocity_obstacle_observation_wrapper import VelocityObstacleObservationWrapper
from environments.wrappers.dynamic_window_action_wrapper import DynamicWindowActionWrapper

def parse_args():
    parser=argparse.ArgumentParser(description='Training script')

    # Testing and debugging
    parser.add_argument('--no_save', action='store_true', default=False, help='Do not save the config, model or logs')

    # Webots settings
    parser.add_argument('--headless', action='store_true', default=False, help='Run Webots in headless mode')
    
    # Environment settings
    parser.add_argument('--envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--env_proto_config', type=str, default='sparse_lidar_proto_config.json', help='Proto config file to adjust proto settings')
    parser.add_argument('--wrapper_classes', nargs='+', default=['SparseLidarObservationWrapper', 'CommandVelocityActionWrapper'], help='List of wrapper classes around BaseEnv that are applied chronologically')
    
    # Model settings
    parser.add_argument('--comment', type=str, default='test', help='Comment that hints the setup used for training')
    parser.add_argument('--policy_type', type=str, default='MlpPolicy', help='Policy type used for configuring the model')
    parser.add_argument('--policy_kwargs', type=dict, default={
        'net_arch': dict(pi=[256, 256, 256, 256, 256], vf=[256, 256, 256, 256, 256]),
        'activation_fn': th.nn.ReLU  # Better for obs normalized between [0, 1]
    }, help='Policy kwargs used for configuring the model')
    parser.add_argument('--use_init_model', action='store_true', default=False, help='Continue training from a saved model (latest model if datetime or steps are not provided)')
    parser.add_argument('--init_model_datetime', type=str, default=None, help='Date and time of the model to continue training from in the format YY_MM_DD__HH_MM_SS')
    parser.add_argument('--init_model_steps', type=int, default=None, help='Number of steps of the model to continue training from')
    
    # Train and callback settings
    parser.add_argument('--steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--model_save_freq', type=int, default=10000, help='Model save frequency')
    parser.add_argument('--model_eval_freq', type=int, default=10000, help='Model evaluation frequency')
    parser.add_argument('--model_n_eval_episodes', type=int, default=25, help='Number of evaluation episodes')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    
    args=parser.parse_args()
    return args

def save_config(args, dir):    
    # Function to get values when simply wanting to continue training from latest model
    if args.use_init_model and (args.init_model_datetime is None or args.init_model_steps is None):
        latest_model_dir=at.get_latest_model_dir()
        if latest_model_dir is None:
            raise FileNotFoundError("No model directory found, continuing training is not possible.")
        args.init_model_datetime=latest_model_dir.split('/')[-1]
        args.init_model_steps=at.get_latest_model_steps(latest_model_dir)
        # TODO: load other parameters from archive to continue training without setting envs, steps, etc.

    # Create config file from args and wrapper+base+proto parameters
    train_args={
        'comment': args.comment,
        'environment': {
            'envs': args.envs,
            'env_proto_config': args.env_proto_config,
            'wrapper_classes': args.wrapper_classes,
        },
        'model': {
            'policy_type': args.policy_type,
            'policy_kwargs': args.policy_kwargs,
            'use_init_model': args.use_init_model,
            'init_model_datetime': args.init_model_datetime,
            'init_model_steps': args.init_model_steps,
        },
        'train_and_callback': {
            'steps': args.steps,
            'model_save_freq': args.model_save_freq,
            'model_eval_freq': args.model_eval_freq,
            'model_n_eval_episodes': args.model_n_eval_episodes,
            'log_interval': args.log_interval,
        }
    }
    wrapper_params = {}
    for wrapper_class in args.wrapper_classes:
        if hasattr(globals()[wrapper_class], 'params_file_name'):
            wrapper_file_name = globals()[wrapper_class].params_file_name 
            wrapper_params.update({wrapper_file_name: at.load_parameters([wrapper_file_name])})
    config = {
        'train_args': train_args,
        'reward_params': at.load_parameters(['parameterized_reward.yaml']),
        'wrapper_params': wrapper_params,
        'proto_params': at.load_parameters([args.env_proto_config]),
        'base_params': at.load_parameters(['base_parameters.yaml']),
    }

    if not args.no_save:
        # Save the config to a yaml file
        os.makedirs(dir, exist_ok=True)
        config_file=os.path.join(dir, "training_config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

def apply_time_limit(env):
    # Add time limit wrapper to limit the episode length without breaking Markov assumption
    params = at.load_parameters(['base_parameters.yaml'])
    max_episode_steps = params['max_ep_time'] / params['sample_time']
    return TimeLimit(env, max_episode_steps=max_episode_steps)

def load_init_model(args):
    if args.init_model_datetime is None or args.init_model_steps is None:
        raise ValueError("When using an initial model, both the datetime and steps must be provided.")
    
    package_dir = os.path.abspath(os.pardir)
    model_dir=os.path.join(package_dir,'training', 'archive', 'models', args.init_model_datetime)
    model_name=f'rl_model_{args.init_model_steps}_steps.zip'
    model_path=os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} does not exist, continuing training is not possible.")

    return PPO.load(model_path)

def train(args, model_dir, log_dir):
    # Environment settings
    envs=args.envs
    env_proto_config=args.env_proto_config
    wrapper_classes=[globals()[wrapper] for wrapper in args.wrapper_classes]
    # Add montior wrapper to know the episode reward, length, time and other data
    wrapper_classes = wrapper_classes + [apply_time_limit] + [Monitor]
    
    # Model settings
    policy_type=args.policy_type

    # Train and callback settings
    steps=args.steps
    model_save_freq=args.model_save_freq
    model_eval_freq=args.model_eval_freq
    model_n_eval_episodes=args.model_n_eval_episodes
    log_interval=args.log_interval

    # Creating vectorized environment
    vec_env=make_vec_env(
        env_id=BaseEnv,
        wrapper_class=lambda env: bt.chain_wrappers(env, wrapper_classes),
        n_envs=envs, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            'proto_config': env_proto_config,
            'wb_headless': args.headless,
        }
    )
    
    # Creating PPO model with callbacks
    if not args.use_init_model:
        model=PPO(
            policy=policy_type,
            env=vec_env,
            tensorboard_log=log_dir,
            policy_kwargs=args.policy_kwargs
        )
    else:
        model=load_init_model(args)
        model.set_env(vec_env)
        model.tensorboard_log=log_dir
    # Disables logging
    if args.no_save:
        model.tensorboard_log=None
    checkpoint_callback=CheckpointCallback(
        save_freq=model_save_freq,
        save_path=model_dir,
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )
    eval_callback=EvalCallback(
        eval_env=vec_env,
        callback_on_new_best=None,
        callback_after_eval=None,
        n_eval_episodes=model_n_eval_episodes,
        eval_freq=model_eval_freq,
        log_path=None,
        best_model_save_path=model_dir,
        deterministic=False,
        render=False,
        verbose=0,
        warn=True,
    )

    # Train model
    if not args.no_save:
        model.learn(
            total_timesteps=steps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=log_interval,
            reset_num_timesteps=False,
            progress_bar=True
        )
    else:
        model.learn(
            total_timesteps=steps,
            log_interval=0,
            reset_num_timesteps=False,
            progress_bar=True
        )

def main():
    # Parse training arguments
    args=parse_args()

    # Directories for bookkeeping
    date, time=at.get_date_time()
    config_dir=f'./archive/configs/{date}__{time}'
    model_dir=f'./archive/models/{date}__{time}'
    log_dir=f'./archive/logs/{date}__{time}'

    # Save args and other configs to yaml file
    save_config(args, config_dir)

    # Train using parameters parsed
    train(args, model_dir, log_dir)
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Kill all webots processes when user interrupts the program
        bt.killall_webots()
        sys.exit(0)