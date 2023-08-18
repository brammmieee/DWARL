# %%
%load_ext autoreload
%autoreload # reloads custom modules into Ipython interpreter
%matplotlib qt

# %%
import os 
import yaml
import tensorrt
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import TimeLimit

import utils.custom_tools as ct
import callbacks.custom_callbacks as cc
from environments.custom_env import CustomEnv

from stable_baselines3.td3 import TD3
from stable_baselines3.td3.policies import MultiInputPolicy
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

np.set_printoptions(precision=5, suppress=True)

package_dir = os.path.abspath(os.pardir)
params_dir = os.path.join(package_dir, 'parameters')
monitor_dir = os.path.join(os.getcwd(), 'monitor')

parameters_file = os.path.join(params_dir, 'parameters.yml')
with open(parameters_file, "r") as file:
    params = yaml.safe_load(file)

# ============================== # Creating the environment # ============================ #
# %% 
env = CustomEnv(render_mode=None, wb_open=True, wb_mode='training')
check_env(env)
env = TimeLimit(env, max_episode_steps=(params['max_ep_time']/params['sample_time']))
env = Monitor(
    env=env, 
    filename=None,
    info_keywords=(),  # can be used for logging the parameters for each test run for instance
)
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
model_name = ct.get_file_name_with_date(test_nr_today=2, comment='_less_maps__goal_prog_rew__new_action')

# policy_kwargs = dict(net_arch=dict(pi=[120, 120, 120], vf=[120, 120, 120]))
# Create the agent
model = PPO(
    policy=MultiInputPolicy,
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




# ================================= # Loading Non-Archived Model # ====================== #
# %% Name
model_name = 'insert model name'
n_steps_load = 00000000

# %% Load model
model = PPO.load('./models' + '/' + model_name +  '/' + model_name + '_' + str(n_steps_load) + '_steps')

# %% Set new env
model.set_env(env=env)






# =================================== # Evaluating policy # ============================= #
# %% 
env = CustomEnv(render_mode='full', wb_open=True, wb_mode='testing', reward_monitoring=False)

# %% Load model
model = PPO.load('./models' + '/insert model name')

# %% Setting render mode and eval vars
env.set_render_mode('full')
nr_eval_eps = 10

# %% Evaluation runs
mean_reward, std_reward = evaluate_policy(
    model, 
    env=env, 
    n_eval_episodes=nr_eval_eps, 
    deterministic=True, 
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
    warn=True,
)
print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')

# ============================= # Reward monitoring # =================================== #
# %%
%matplotlib qt

# %% 
from environments.custom_env import CustomEnv
from utils import custom_tools as ct
import numpy as np
import matplotlib.pyplot as plt

# %% 
env = CustomEnv(render_mode='trajectory', wb_open=True, wb_mode='testing', reward_monitoring=True)
nr_eps = 1

# %%
ep_reward_list = []
ep_reward = 0
obs = env.reset()
while True:
    # action, _ = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    action = np.array([1.0, -1.0])

    # assert env.action_space.contains(action)    
    # print(f'action = {action}')

    obs, reward, done, info = env.step(action, teleop=True)
    # print(f'reward = {reward}')
    ep_reward += reward

    if done:
        obs = env.reset()
        ep_reward_list.append(ep_reward)
        ep_reward = 0
        if len(ep_reward_list) >= nr_eps:
            break 

# %%
# NOTE: use the file_name of with the max ep idx 
reward_matrix = ct.read_pickle_file(file_name='rewards_ep_1', file_dir=os.path.join('training','rewards'))

fig, ax = plt.subplots()
ax.set_xlabel('timestep')
ax.set_ylabel('reward')
ax.grid()

for ep_rewards in reward_matrix[:]:
    ax.plot(list(range(0, len(ep_rewards),1)), ep_rewards)

ax.legend([f'ep_{ep_nr+1}' for ep_nr in range(len(reward_matrix))])

# ============================ # Debugging # ============================================= #
# %%
%matplotlib qt

# %% 
from environments.custom_env import CustomEnv
import numpy as np

# %%
env = CustomEnv(render_mode='full', wb_open=True, wb_mode='testing', reward_monitoring=False)

# %%
obs = env.reset() #options={"map_nr":40, "nominal_dist":1})

# %%
action = np.array([1.0, 1.0])
obs, reward, done, info = env.step(action, teleop=False)

# %% 
import time

# env.set_render_mode('velocity') 
# obs = env.reset()
n_steps = 100000000
for i in range(n_steps):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action, teleop=True)
    # time.sleep(0.4)
    if done:
        obs = env.reset()

# =========================== # Changing map resolution # ============================= #
# %%
from utils.map2proto import Map2ProtoConverter

# %%
m2p = Map2ProtoConverter()

# %%
m2p.convert(map_res = params['map_res'])

# =========================== # Creating training map lists # ========================= #
# %% Create the training and test map lists
train_map_nr_list, test_map_nr_list = ct.sample_training_test_map_nrs(range_start=0, range_end=299, training_ratio=0.7)

# %% Save map lists to pickle files #NOTE: proceed with caution, files will be overwritten
ct.write_pickle_file('train_map_nr_list', 'parameters', train_map_nr_list)
ct.write_pickle_file('test_map_nr_list', 'parameters', test_map_nr_list)

# %% Check if everything went well
train_map_nr_list = ct.read_pickle_file('train_map_nr_list', 'parameters')
test_map_nr_list = ct.read_pickle_file('test_map_nr_list', 'parameters')
print(f'train_map_nr_list = {train_map_nr_list}')
print(f'test_map_nr_list = {test_map_nr_list}')
