# %%
%load_ext autoreload
# %autoreload 2 # reloads custom modules into Ipython interpreter
%matplotlib qt

# %%
import os
import yaml
import numpy as np
from gym.wrappers import TimeLimit

import utils.admin_tools as at
from environments.base_env import BaseEnv

from stable_baselines3.td3 import TD3
from stable_baselines3.td3.policies import MultiInputPolicy
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from environments.wrappers.sparse_lidar_observation_wrapper import SparseLidarObservationWrapper as SLObsWrapper
from environments.wrappers.command_velocity_action_wrapper import CommandVelocityActionWrapper as CVActWrapper
from environments.wrappers.parameterized_reward_wrapper import ParameterizedRewardWrapper as PrewWrapper

np.set_printoptions(precision=5, suppress=True)

# ============================== # Creating the environment # ============================ #

# %% Single envS
# env = BaseEnv(render_mode=None, wb_open=True, wb_mode='training')
env = PrewWrapper(CVActWrapper(SLObsWrapper(BaseEnv(render_mode=None, wb_open=True, wb_mode='training', proto_config='sparse_lidar_proto_config.json'))))

# env = TimeLimit(env, max_episode_steps=(params['max_ep_time']/params['sample_time']))
# env = Monitor(
#     env=env,
#     filename=None,
#     info_keywords=(),  # can be used for logging the parameters for each test run for instance
# )
# vec_env = DummyVecEnv([lambda: env])
# vec_env = VecNormalize(
#     venv=vec_env,
#     training=True,
#     norm_obs=True,
#     norm_reward=True,
#     clip_obs=10.0,
#     clip_reward=10.0,
#     gamma=0.99,
#     epsilon=1e-8,
#     norm_obs_keys=None,
# )

# ====================================== # Training # ==================================== #
# %% Model #NOTE: alternatively run the "Loading non-archived model section!"
model_name = at.get_file_name_with_date(test_nr_today=0, comment='01_09_24_test')

# policy_kwargs = dict(net_arch=dict(pi=[120, 120, 120], vf=[120, 120, 120]))
# Create the agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    tensorboard_log = "./logs/" + model_name,
    # policy_kwargs = policy_kwargs,
    # learning_rate= 1e-4,
    # n_steps = 4000, # increase the
    # batch_size = 500, # increase batch size
    # n_epochs = 10,
    # gamma = 0.999, # more emphasis on future rewards
    # ent_coef = 0.01, # increase exploration
)

# %% Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq = 10000,
    save_path = "./models/" + model_name,
    name_prefix = model_name,
    save_replay_buffer = False,
    save_vecnormalize = False,
    verbose = 0,
)
# stop_training_callback = StopTrainingOnNoModelImprovement(
#     max_no_improvement_evals = 10,
#     min_evals = 25,
#     verbose = 1,
# )
eval_callback = EvalCallback(
    eval_env = env,
    callback_on_new_best = None,
    callback_after_eval = None,
    n_eval_episodes = 25,
    eval_freq = 15000,
    log_path = None,
    best_model_save_path = "./models",
    deterministic = False,
    render = False,
    verbose = 0,
    warn = True,
)
callback_list = CallbackList([checkpoint_callback, eval_callback]) #NOTE: can also pass list directly to learn

# %% Train model
model.learn(
    total_timesteps=1e8,
    callback=callback_list,
    log_interval=10,
    tb_log_name=model_name,
    reset_num_timesteps=False,
    progress_bar=True
)

# %%