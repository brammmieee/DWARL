# %%
import numpy as np
import matplotlib.pyplot as plt

nobleo_colors = {'purple': np.array([154, 71, 153]),
                 'blue': np.array([0, 102, 225]),
                 'orange': np.array([246, 164, 37]),
                 'lightblue': np.array([23,133,194])}

class Plotter():
    def __init__(self, eval_results):
        self.done_cause_colors = {'at_goal': 'lightgreen',
                                  'outside_map': 'skyblue',
                                  'collision': nobleo_colors['orange']/255,
                                  'max_nr_steps_reached': 'lightcoral'}
        self.eval_results = eval_results
        self.nr_maps = len(eval_results)
        self.max_axes_per_figure = 25

    def initialize_figure(self):
        self.figs = {}
        map_inds = np.arange(0, self.nr_maps, self.max_axes_per_figure)
        map_inds = np.append(map_inds, self.nr_maps)
        nr_axes = map_inds[1] - map_inds[0]

        self.legend_elements = []
        for key, value in self.done_cause_colors.items():
            self.legend_elements.append(plt.Line2D([0], [0], color=value, lw=4, label=key))

        self.nr_figs = len(map_inds)-1
        for fig_ind in range(self.nr_figs):
            nr_rows = int(np.ceil(np.sqrt(nr_axes)))
            nr_cols = int(np.ceil(nr_axes / nr_rows))
            fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(15, 15))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])  # Necessary if nr_rows = nr_cols = 1
            for i, map_ind in enumerate(range(map_inds[fig_ind], map_inds[fig_ind+1])):
                self.figs[map_ind] = {'fig_ind': fig_ind,'fig': fig, 'ax': axes.flat[i]}
            fig.legend(handles=self.legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)


    def plot_grid(self, eval_result, ax):
        ax.set_aspect('equal', adjustable='box')
        xlim = [0,0]
        ylim = [0,0]
        for box in eval_result['map']:
            min_x = min(vertex[0] for vertex in box)
            min_y = min(vertex[1] for vertex in box)
            width = max(vertex[0] for vertex in box) - min_x
            height = max(vertex[1] for vertex in box) - min_y
            xlim[0] = min(xlim[0], min_x)
            xlim[1] = max(xlim[1], min_x + width)
            ylim[0] = min(ylim[0], min_y)
            ylim[1] = max(ylim[1], min_y + height)
            rect = plt.Rectangle((min_x, min_y), width, height, facecolor=nobleo_colors['purple']/255, edgecolor=nobleo_colors['purple']/255)
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
        ax.set_facecolor(self.done_cause_colors[done_cause])
