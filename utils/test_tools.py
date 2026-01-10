from pathlib import Path
import json
import numpy as np
import os
import matplotlib.pyplot as plt
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy


nobleo_colors = {'purple': np.array([154, 71, 153]),
                 'blue': np.array([0, 102, 225]),
                 'orange': np.array([246, 164, 37]),
                 'lightblue': np.array([23,133,194])}

def evaluate_model(nr_episodes, env, model, max_nr_steps, deterministic=False, seed=0):
    """ Evaluation of the model inspired by the evaluate function from Sb3 """
    results = []
    for i in range(nr_episodes):
        print(f'Evaluating episode {i}/{nr_episodes}')
        result = evaluate_single_map(env, model, max_nr_steps, deterministic, seed)
        results.append(result)
    return results
        
def evaluate_single_map(env, model, max_nr_steps, deterministic=False, seed=0):
    # Initialize
    # env = Monitor(env)      # NOTE: See sb3 evaluate, is this necessary?
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
        'max_nr_steps': max_nr_steps,
        'observations': [],
        'local_goal_pos': [],
        'actions': []
    }

    result['positions'].append(env.unwrapped.cur_pos)
    result['orientations'].append(env.unwrapped.cur_orient_matrix)
    result['velocities'].append(env.unwrapped.cur_vel)
    result['local_goal_pos'].append(env.unwrapped.local_goal_pos)
    
    states = None
    for _ in range(max_nr_steps):
        action, states = model.predict(
            obs,
            state=states,
            episode_start=None,   # This parameter is only used for reinforcement learning
            deterministic=deterministic
        )
        obs, reward, done, _, _ = env.step(action)
        
        result['positions'].append(env.unwrapped.cur_pos)
        result['orientations'].append(env.unwrapped.cur_orient_matrix)
        result['velocities'].append(env.unwrapped.cur_vel)
        result['rewards'].append(reward)
        result['observations'].append(obs)
        result['local_goal_pos'].append(env.unwrapped.local_goal_pos)
        result['actions'].append(action)
                    
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
        # self.nr_figs = 0

    def plot_paths(self, results, block=False):
        # Plot the traversed paths of all maps in 1 or more figures with multiple axes
        nr_maps = len(results)
        self.create_figures(nr_maps, results)
        for i, result in enumerate(results):
            ax = self.figs[i]['ax']
            self.plot_grid(result, ax)
            self.plot_traversed_path(result, ax)
        self.set_legend(results)
        if self.cfg.show:
            plt.show(block=block)

    def plot_velocities(self, results, block=False):
        # Plot the absolute velocities over time for all maps in 1 or more figures with multiple axes
        nr_maps = len(results)
        self.create_figures(nr_maps, results)
        for i, result in enumerate(results):
            ax = self.figs[i]['ax']
            self.plot_velocity(result, ax)
        self.set_legend(results)

        if self.cfg.show:
            plt.show(block=block)

    def plot_velocity(self, eval_result, ax):
        # Plot velocity profile in axes ax
        linear_velocity = [velocity[1] for velocity in eval_result['velocities']]
        ax.plot(linear_velocity, color=nobleo_colors['purple']/255)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Forward velocity')
        ax.set_title(eval_result['map_name'])
        ax.set_facecolor(self.cfg.done_cause_colors[eval_result['done_cause']])

    def create_figures(self, nr_maps, results):
        # Create all figures and axes
        self.nr_figs = (nr_maps - 1) // self.cfg.max_nr_axes + 1
        map_inds = list(range(0, nr_maps, self.cfg.max_nr_axes)) + [nr_maps]

        for fig_ind in range(self.nr_figs):
            nr_axes = map_inds[fig_ind+1] - map_inds[fig_ind]
            nr_rows = int(np.ceil(np.sqrt(nr_axes)))
            nr_cols = int(np.ceil(nr_axes / nr_rows))
            fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(15, 15))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])  # Necessary if nr_rows = nr_cols = 1
            for i, map_ind in enumerate(range(map_inds[fig_ind], map_inds[fig_ind+1])):
                self.figs[map_ind] = {'fig_ind': fig_ind,'fig': fig, 'ax': axes.flat[i]}

    def get_legend_elements(self, fig_ind):
        # Get legend elements for a figure
        legend_elements = []
        for key, color in self.cfg.done_cause_colors.items():
            label = f"{key}: {self.done_cause_count[fig_ind][key]}/{self.total_done_cause_count[key]}"
            legend_element = plt.Line2D([0], [0], color=color, lw=4, label=label)
            legend_elements.append(legend_element)
        return legend_elements
    
    def set_legend(self, results):
        # Set the legend for all figures including the number of done causes per figure and in total
        self.count_done_causes(results)

        fig_inds = []
        for map_ind in range(len(self.figs)):
            if self.figs[map_ind]['fig_ind'] not in fig_inds:
                fig_ind = self.figs[map_ind]['fig_ind']
                fig_inds.append(fig_ind)
                fig_handle = self.figs[map_ind]['fig']
                legend_elements = self.get_legend_elements(fig_ind)
                fig_handle.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)

    def count_done_causes(self, results):
        # Count the occurence of each done cause per figure and in total over all maps
        done_cause_count = {}  # Count of done causes for each figure
        total_done_cause_count = {}  # Total count of done causes over all figures
        for done_cause in self.cfg.done_cause_colors.keys():
            total_done_cause_count[done_cause] = 0
        for map_ind in range(len(results)):
            total_done_cause_count[results[map_ind]['done_cause']] += 1
            self.total_done_cause_count = total_done_cause_count

        # Per figure:
        if hasattr(self, 'nr_figs'):
            for fig_ind in range(self.nr_figs):
                done_cause_count[fig_ind] = {}
                for done_cause in self.cfg.done_cause_colors.keys():
                    done_cause_count[fig_ind][done_cause] = 0
            for map_ind in range(len(self.figs)):
                fig_ind = self.figs[map_ind]['fig_ind']
                done_cause_count[fig_ind][results[map_ind]['done_cause']] += 1
            self.done_cause_count = done_cause_count

    def plot_grid(self, eval_result, ax):
        # Plot the map grid in axes ax
        ax.set_aspect('equal', adjustable='box')
        xlim = [np.nan, np.nan]
        ylim = [np.nan, np.nan]
        for box in eval_result['map']:
            min_x = min(vertex[0] for vertex in box)
            min_y = min(vertex[1] for vertex in box)
            width = max(vertex[0] for vertex in box) - min_x
            height = max(vertex[1] for vertex in box) - min_y
            xlim[0] = np.nanmin([xlim[0], min_x])
            xlim[1] = np.nanmax([xlim[1], min_x + width])
            ylim[0] = np.nanmin([ylim[0], min_y])
            ylim[1] = np.nanmax([ylim[1], min_y + height])
            rect = plt.Rectangle((min_x, min_y), width, height, facecolor=nobleo_colors['purple']/255, edgecolor=nobleo_colors['purple']/255)
            ax.add_patch(rect)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(eval_result['map_name'])

    def plot_traversed_path(self, eval_result, ax):
        # Plot the traversed path in axes ax
        positions = np.array(eval_result['positions'])
        init_pose = eval_result['init_pose']
        goal_pose = eval_result['goal_pose']
        done_cause = eval_result['done_cause']
        ax.plot(init_pose[0], init_pose[1], 'go')
        ax.plot(goal_pose[0], goal_pose[1], 'ro')
        ax.plot(positions[:,0], positions[:, 1], color='k')
        ax.set_facecolor(self.cfg.done_cause_colors[done_cause])
        self.set_map_limits(ax, eval_result)

    def set_map_limits(self, ax, eval_result):

        # X limits
        xlim = ax.get_xlim()
        positions = np.array(eval_result['positions'])
        path_x_min = np.min(positions[:,0])
        path_x_max = np.max(positions[:,0])
        path_x_dist = path_x_max - path_x_min
        x_margin = 0.1 * path_x_dist
        xlim_min = min(xlim[0], path_x_min - x_margin, eval_result['init_pose'][0] - x_margin, eval_result['goal_pose'][0] - x_margin)
        xlim_max = max(xlim[1], path_x_max + x_margin, eval_result['init_pose'][0] + x_margin, eval_result['goal_pose'][0] + x_margin)
        ax.set_xlim(xlim_min, xlim_max)

        # Y limits
        ylim = ax.get_ylim()
        path_y_min = np.min(positions[:,1])
        path_y_max = np.max(positions[:,1])
        path_y_dist = path_y_max - path_y_min
        y_margin = 0.1 * path_y_dist
        ylim_min = min(ylim[0], path_y_min - y_margin, eval_result['init_pose'][1] - y_margin, eval_result['goal_pose'][1] - y_margin)
        ylim_max = max(ylim[1], path_y_max + y_margin, eval_result['init_pose'][1] + y_margin, eval_result['goal_pose'][1] + y_margin)
        ax.set_ylim(ylim_min, ylim_max)

    def save_plots(self, output_folder, prefix='figure'):
        # Save the specific figures to png files
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for fig_ind in range(self.nr_figs):
            for fig in self.figs.values():
                if fig['fig_ind'] == fig_ind:
                    png_file_path = os.path.join(output_folder, f'{prefix}_{fig_ind}.png')
                    print(f'Saving figure to:              {png_file_path}') 
                    fig['fig'].savefig(png_file_path, bbox_inches='tight')
                    break


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_results(eval_result, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(eval_result, f, indent=4, cls=NumpyEncoder)

def load_eval_results(json_file_path):
    with open(json_file_path, 'r') as f:
        eval_result = json.load(f)
    return eval_result




