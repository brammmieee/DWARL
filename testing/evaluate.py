#!/usr/bin/env python3

import os
import argparse
from environments.base_env import BaseEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO
from environments.wrappers.sparse_lidar_observation_wrapper import SparseLidarObservationWrapper
from environments.wrappers.command_velocity_action_wrapper import CommandVelocityActionWrapper
from environments.wrappers.velocity_obstacle_observation_wrapper import VelocityObstacleObservationWrapper
from environments.wrappers.dynamic_window_action_wrapper import DynamicWindowActionWrapper
from environments.wrappers.parameterized_reward_wrapper import ParameterizedRewardWrapper

import utils.admin_tools as at
import utils.base_tools as bt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate trained model')

    # Obligatory arguments
    parser.add_argument('--date', type=str, help='Training date in the format YY_MM_DD', required=True)
    parser.add_argument('--time', type=str, help='Training time in the format HH_MM_SS', required=True)

    # Optional arguments
    parser.add_argument('--steps', type=int, default=None, help='Number of training steps')
    parser.add_argument('--evaluation_episodes', type=int, default=10, help='Number of episodes to evaluate the model')
    parser.add_argument('--deterministic', type=bool, default=False, help='Whether to use deterministic policy or not')
    parser.add_argument('--webots_mode', type=str, default='testing', help="'testing' or 'training'")
    parser.add_argument('--render_mode', type=str, default=None, help="'trajectory', 'full', 'velocity', default to no rendering")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Model Name
    training_date = args.date
    training_time = args.time
    training_steps = args.steps

    evaluation_episodes = args.evaluation_episodes
    deterministic = args.deterministic
    webots_mode = args.webots_mode
    render_mode = args.render_mode

    # Datetime
    date_time = f'{training_date}__{training_time}'
    package_dir = os.path.abspath(os.pardir)


    # Load the training config file
    training_config = at.load_parameters(
        file_name_list='training_config.yaml',
        start_dir=os.path.join(package_dir, f'training/archive/configs/{date_time}')
    )

    # Recreate env used during training
    wrapper_classes=[globals()[wrapper] for wrapper in training_config['environment']['wrapper_classes']]
    base_env = BaseEnv(
        render_mode=render_mode, 
        wb_open=True, 
        wb_mode=webots_mode,
        proto_config=training_config['environment']['env_proto_config']
    )
    env = bt.chain_wrappers(base_env, wrapper_classes)

    # Load model
    prefix = os.path.join(package_dir, f'training/archive/models/{date_time}/')
    if training_steps:
        # Load model based on training steps
        model = PPO.load(prefix + f'rl_model_{training_steps}')
    else:
        # Load best model
        model = PPO.load(prefix + 'best_model', env=env)
    model.set_env(env=env)

    # Evaluation runs
    mean_reward, std_reward = evaluate_policy(
        model,
        env=env,
        n_eval_episodes=evaluation_episodes,
        deterministic=deterministic,
        render=False,
        callback=None,
        reward_threshold=None,
        return_episode_rewards=False,
        warn=True,
    )

    # Print results
    print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')

if __name__ == '__main__':
    main()