import json
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_model(nr_episodes, env, model, max_nr_steps, deterministic=False, seed=0):
    """ Evaluation of the model inspired by the evaluate function from Sb3 """
    results = []
    for _ in range(nr_episodes):
        result = evaluate_single_map(env, model, max_nr_steps, deterministic, seed)
        results.append(result)
    return results
        
def evaluate_single_map(env, model, max_nr_steps, deterministic=False, seed=0):
    # Initialize
    env = Monitor(env)      # NOTE: See sb3 evaluate, is this necessary?
    obs, _ = env.reset()
    result = {
        'map_name': env.unwrapped.map_name,
        'map': env.unwrapped.map,
        'init_pose': env.unwrapped.init_pose,
        'goal_pose': env.unwrapped.goal_pose,
        'positions': [],
        'orientations': [],
        'velocities': [],
        'rewards': [],
        'done_cause': None,
        'max_nr_steps': max_nr_steps
    }
    
    result['positions'].append(env.unwrapped.cur_pos)
    result['orientations'].append(env.unwrapped.cur_orient_matrix)
    result['velocities'].append(env.unwrapped.cur_vel)
    
    states = None
    for _ in range(max_nr_steps):
        action, states = model.predict(
            obs,
            state=states,      # NOTE: How does this work?
            episode_start=1,   # NOTE: Why is this 1?
            deterministic=deterministic
        )
        obs, reward, done, _, _ = env.step(action)
        
        result['positions'].append(env.unwrapped.cur_pos)
        result['orientations'].append(env.unwrapped.cur_orient_matrix)
        result['velocities'].append(env.unwrapped.cur_vel)
        result['rewards'].append(reward)
                    
        if done:
            result['done_cause'] = env.unwrapped.done_cause
            result['nr_steps'] = len(result['positions'])
            break
        
    if not done:
        result['done_cause'] = 'max_nr_steps_reached'
        result['nr_steps'] = max_nr_steps
    
    return result

class ResultPlotter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.figs = {}
        self.nr_figs = 0

    def plot_results(self, results, output_folder=None):
        nr_maps = len(results)
        self.create_figures(nr_maps)
        
        for i, result in enumerate(results):
            ax = self.figs[i]['ax']
            self.plot_grid(result, ax)
            self.plot_traversed_path(result, ax)

        if self.cfg.show:
            plt.show()
        
        if self.save_plots:
            if output_folder is None:
                print('No output folder specified. Figures will not be saved.')
            self.save_plots_(output_folder)

    def create_figures(self, nr_maps):
        self.nr_figs = (nr_maps - 1) // self.cfg.max_axis + 1
        map_inds = list(range(0, nr_maps, self.cfg.max_axis)) + [nr_maps]
        
        legend_elements = [plt.Line2D([0], [0], color=value, lw=4, label=key) 
                           for key, value in self.cfg.done_cause_colors.items()]
        
        for fig_ind in range(self.nr_figs):
            fig, axes = self.create_subplot(self.cfg.max_axis)
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
            
            for i, map_ind in enumerate(range(map_inds[fig_ind], map_inds[fig_ind+1])):
                self.figs[map_ind] = {'fig_ind': fig_ind, 'fig': fig, 'ax': axes.flat[i]}

    def create_subplot(self, nr_axes):
        nr_rows = int(np.ceil(np.sqrt(nr_axes)))
        nr_cols = int(np.ceil(nr_axes / nr_rows))
        fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(15, 15))
        return fig, np.array([axes]) if not isinstance(axes, np.ndarray) else axes

    def plot_grid(self, eval_result, ax):
        ax.set_aspect('equal', adjustable='box')
        xlim = [0, 0]
        ylim = [0, 0]
        for box in eval_result['map']:
            min_x = min(vertex[0] for vertex in box)
            min_y = min(vertex[1] for vertex in box)
            width = max(vertex[0] for vertex in box) - min_x
            height = max(vertex[1] for vertex in box) - min_y
            xlim[0] = min(xlim[0], min_x)
            xlim[1] = max(xlim[1], min_x + width)
            ylim[0] = min(ylim[0], min_y)
            ylim[1] = max(ylim[1], min_y + height)
            rect = plt.Rectangle((min_x, min_y), width, height, facecolor='purple', edgecolor='purple')
            ax.add_patch(rect)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def plot_traversed_path(self, eval_result, ax):
        positions = np.array(eval_result['positions'])
        init_pose = eval_result['init_pose']
        goal_pose = eval_result['goal_pose']
        done_cause = eval_result['done_cause']
        ax.plot(init_pose[0], init_pose[1], 'go')
        ax.plot(goal_pose[0], goal_pose[1], 'ro')
        ax.plot(positions[:,0], positions[:, 1], color='k')
        ax.set_facecolor(self.cfg.done_cause_colors[done_cause])

    def save_plots_(self, output_folder):
        for fig_ind in range(self.nr_figs):
            for fig in self.figs.values():
                if fig['fig_ind'] == fig_ind:
                    png_file_path = os.path.join(output_folder, f'figure_{fig_ind}.png')
                    fig['fig'].savefig(png_file_path, bbox_inches='tight')
                    break




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_eval_result(eval_result, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(eval_result, f, indent=4, cls=NumpyEncoder)

def load_eval_result(json_file_path):
    with open(json_file_path, 'r') as f:
        eval_result = json.load(f)
    return eval_result




