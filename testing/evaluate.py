# %%
from environments.custom_env import CustomEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import PPO

# %% Model Name
model_name = 'insert model name'
n_steps_load = 00000000

# Create env and load model
env = CustomEnv(render_mode='full', wb_open=True, wb_mode='testing', reward_monitoring=False)
model = PPO.load('./models' + '/' + model_name +  '/' + model_name + '_' + str(n_steps_load) + '_steps')
model.set_env(env=env)

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