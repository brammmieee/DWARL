# %%
%matplotlib qt

%load_ext autoreload
%autoreload 2

import hydra
from omegaconf import DictConfig
import sys

from environments.base_env import BaseEnv
from environments.webots_env import WebotsEnv
from stable_baselines3.ppo import PPO

from utils.test_tools import evaluate_model, extract_eval_result, plot_eval_results, save_eval_results, save_figures, load_eval_results
from utils.plot_tools import nobleo_colors, Plotter
# # import numpy as np
# # import json
from utils.data_loader import InfiniteDataLoader
import utils.data_set as ds
from utils.wrapper_tools import wrap_env


# %% 
# Choose the netwerk to evaluate
date_time = '24_09_24__17_24_45'
training_steps = 92320000  # Set the number of training steps (optional)

# Evaluation parameters
deterministic = False  # Set whether to use deterministic policy or not
nr_maps = 25
max_nr_steps = 1000  # Evaluation will stop after this number of steps

# %% Create model and env

#  --- Only for jupyter notebook! ---
sys.argv = [arg for arg in sys.argv if not arg.startswith('--f=')]
hydra.initialize(config_path='DWARL/config', version_base='1.1')
cfg = hydra.compose(config_name='train')
# %% --- End ---

# @hydra.main(config_path='DWARL/config', config_name='train', version_base='1.1')
# def main(cfg: DictConfig):

data_set = ds.Dataset(cfg.paths)
infinite_loader = InfiniteDataLoader(data_set, cfg.envs)
env=BaseEnv(
    cfg=cfg.environment,
    paths=cfg.paths,
    sim_env=WebotsEnv(cfg.simulation, cfg.paths),
    data_loader=infinite_loader,
    env_idx=0,
    render_mode=None,
    evaluate=True
)
env = wrap_env(env, cfg.wrappers)
# main()
# %% Load model
# print(cfg.paths.outputs.models.keys())
# TODO: Adapt to Hydra
import os
package_dir = os.path.abspath(os.pardir)
# prefix = os.path.join(package_dir, f'training/archive/models/{date_time}/')
prefix = '/DWARL/training/archive/models/24_09_24__17_24_45/'
if training_steps:
    # Load model based on training steps
    model = PPO.load(prefix + f'rl_model_{training_steps}_steps', env=env)
else:
    # Load best model
    model = PPO.load(prefix + 'best_model', env=env)


# %% Run evaluation script
eval_results = []
deterministic = False
for i in range(nr_maps):
    print(i)
    env = evaluate_model(model, env, max_nr_steps, deterministic)
    eval_result = extract_eval_result(env, max_nr_steps)
    eval_results.append(eval_result)

# %%
plotter = plot_eval_results(eval_results)

# %%
# Save figures
# TODO: Adapt to Hydra
output_folder = os.path.join(package_dir, 'DWARL', 'testing', 'results', f'{date_time}')
save_figures(plotter, output_folder)

# %% Save evaluation results
# TODO: Adapt to Hydra
output_folder = os.path.join(package_dir, 'DWARL', 'testing', 'results', f'{date_time}')
os.makedirs(output_folder, exist_ok=True)
json_file_path = os.path.join(output_folder, f'eval_results_{training_steps}_steps.json')
save_eval_results(eval_results, json_file_path)

# %% Load evaluation results
# TODO: Adapt to Hydra
date_time = '24_09_24__17_24_45'
training_steps = 92320000  # Set the number of training steps (optional)
package_dir = os.path.abspath(os.pardir)
output_folder = os.path.join(package_dir, 'DWARL', 'testing', 'results', f'{date_time}')
json_file_path = os.path.join(output_folder, f'eval_results_{training_steps}_steps.json')
eval_results2 = load_eval_results(json_file_path)
plot_eval_results(eval_results2)
# %%