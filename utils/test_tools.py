from utils.plot_tools import Plotter
import json
import numpy as np
import os

def evaluate_model(model, env, max_nr_steps, deterministic=False):
    states = None
    observations, info = env.reset()
    done = False
    nr_steps = 0
    while not done and nr_steps < max_nr_steps:
        actions, states = model.predict(
        observations,  # type: ignore[arg-type]
        state=states,
        episode_start=1,
        deterministic=deterministic,
        )
        observations, rewards, done, _, info = env.step(actions)
        nr_steps += 1
    env.nr_steps = nr_steps
    if not done and nr_steps == max_nr_steps:
        env.done_cause = 'max_nr_steps_reached'
    return env

def extract_eval_result(env, max_nr_steps=None):
    eval_result = {}
    attribute_names = [  #'params',
                    #    'grid',
                       'map',
                       'init_pose',
                       'goal_pose',
                       'positions',
                       'done_cause',
                       'velocities',
                       'orientations',
                       'nr_steps',]
    for attribute_name in attribute_names:
        eval_result[attribute_name] = getattr(env, attribute_name)
    eval_result['max_nr_steps'] = max_nr_steps

    return eval_result

def plot_eval_results(eval_results):
    plotter = Plotter(eval_results)
    plotter.initialize_figure()
    for i, eval_result in enumerate(plotter.eval_results):
        plotter.plot_grid(eval_result, plotter.figs[i]['ax'])
        plotter.plot_traversed_path(eval_result, plotter.figs[i]['ax'])
    return plotter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_eval_results(eval_results, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(eval_results, f, indent=4, cls=NumpyEncoder)

def load_eval_results(json_file_path):
    with open(json_file_path, 'r') as f:
        eval_results = json.load(f)
    return eval_results

def save_figures(plotter, output_folder):
    for fig_ind in range(plotter.nr_figs):
        for fig in plotter.figs.values():
            if fig['fig_ind'] == fig_ind:
                png_file_path = os.path.join(output_folder, f'figure_{fig_ind}.png')
                fig['fig'].savefig(png_file_path, bbox_inches='tight')
                break