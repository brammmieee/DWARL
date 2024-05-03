# %%
%matplotlib qt

# %%
import os
from environments.base_env import CustomEnv
from utils import admin_tools as at
import numpy as np
from stable_baselines3.ppo import PPO
import matplotlib.pyplot as plt

# %% env
env = CustomEnv(render_mode='trajectory', wb_open=True, wb_mode='testing', reward_monitoring=True)
nr_eps = 1

# %% model
model_name = 'insert model name'
n_steps_load = 00000000

model = PPO.load('./models' + '/' + model_name +  '/' + model_name + '_' + str(n_steps_load) + '_steps')
model.set_env(env=env)

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
reward_matrix = at.read_pickle_file(file_name='rewards_ep_1', file_dir=os.path.join('training','rewards'))

fig, ax = plt.subplots()
ax.set_xlabel('timestep')
ax.set_ylabel('reward')
ax.grid()

for ep_rewards in reward_matrix[:]:
    ax.plot(list(range(0, len(ep_rewards),1)), ep_rewards)

ax.legend([f'ep_{ep_nr+1}' for ep_nr in range(len(reward_matrix))])