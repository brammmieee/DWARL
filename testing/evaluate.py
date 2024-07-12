# %%
%matplotlib qt

# %%
import os
from environments.base_env import BaseEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO
from environments.wrappers.sparse_lidar_observation_wrapper import SparseLidarObservationWrapper
from environments.wrappers.command_velocity_action_wrapper import CommandVelocityActionWrapper
from environments.wrappers.velocity_obstacle_observation_wrapper import VelocityObstacleObservationWrapper
from environments.wrappers.dynamic_window_action_wrapper import DynamicWindowActionWrapper
import matplotlib.pyplot as plt

import utils.admin_tools as at
import utils.base_tools as bt

# %% Set the following parameters
date_time = '24_07_09__15_06_25'
training_steps = 2940000  # Set the number of training steps (optional)

evaluation_episodes = 25  # Set the number of episodes to evaluate the model
deterministic = False  # Set whether to use deterministic policy or not
webots_mode = 'testing'  # Set the Webots mode ('testing' or 'training')
render_mode = 'full'  # Set the render mode ('trajectory', 'full', 'velocity') or leave it as None for no rendering

# Datetime
package_dir = os.path.abspath(os.pardir)

# %% Run the evaluation script
# Load the training config file
training_config = at.load_parameters(
    file_name_list='training_config.yaml',
    start_dir=os.path.join(package_dir, f'training/archive/configs/{date_time}')
)

# Recreate env used during training
wrapper_classes=[globals()[wrapper] for wrapper in training_config['train_args']['environment']['wrapper_classes']]
base_env = BaseEnv(
    render_mode=render_mode, 
    wb_open=True, 
    wb_mode=webots_mode,
    proto_config=training_config['train_args']['environment']['env_proto_config']
)
env = bt.chain_wrappers(base_env, wrapper_classes)

# Load model
prefix = os.path.join(package_dir, f'training/archive/models/{date_time}/')
if training_steps:
    # Load model based on training steps
    model = PPO.load(prefix + f'rl_model_{training_steps}_steps', env=env)
else:
    # Load best model
    model = PPO.load(prefix + 'best_model', env=env)
# model.set_env(env=env)

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

bt.killall_webots()
plt.close('all')

# Print results
print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')

# %%