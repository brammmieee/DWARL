import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class ParameterizedReward(gym.RewardWrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        
        self.cfg = cfg
        self.done_rewards = self.cfg.done
        self.running_rewards = self.cfg.running

        # Reward plotting
        if self.cfg.render.enabled:
            self.reward_buffers = {component: [] for component in self.running_rewards.keys()}
            self.reward_buffers['total'] = []
            
            # Use the style map from the configuration
            self.reward_style_map = self.cfg.render.style_map
            
            # Use the plot map from the configuration
            self.reward_plot_map = self.cfg.render.plot_map
            
            # Initialize the plot
            self.render_init_plot()

    def reward(self, reward):
        reward = self.get_reward(self.unwrapped.done, self.unwrapped.done_cause)
        if self.cfg.render.enabled:
            self.render()
        return reward

    def get_reward(self, done, done_cause):
        # If the episode is done, return the done reward
        if done:
            done_reward = self.done_rewards[done_cause]
            if self.cfg.render.enabled:
                self.update_reward_buffers('total', done_reward)
            return done_reward
        # Otherwise, calculate the running reward
        else:
            running_reward = 0
            for fn_name, coeff in self.running_rewards.items():
                if coeff == 0:
                    continue
                component_reward = getattr(self, f"calculate_{fn_name}_reward")() * coeff
                running_reward += component_reward
                if self.cfg.render.enabled:
                    self.update_reward_buffers(fn_name, component_reward)
            if self.cfg.render.enabled:
                self.update_reward_buffers('total', running_reward)
            return running_reward
        
    def calculate_linear_reward(self):
        """ NOTE: naming convention for reward calculation functions: calculate_{reward_name}_reward """
        # Calculate the progress difference between the previous and current step
        prev_dist_to_goal = np.linalg.norm(self.unwrapped.goal_pose[:2] - self.unwrapped.prev_pos[:2])
        current_dist_to_goal = np.linalg.norm(self.unwrapped.goal_pose[:2] - self.unwrapped.cur_pos[:2])
        progress_diff = prev_dist_to_goal - current_dist_to_goal
        
        # Normalize the progress difference using the maximum possible progress difference
        v_max = self.unwrapped.cfg.vehicle.kinematics.v_max
        sample_time = self.unwrapped.sim_env.sample_time
        max_progress_diff = v_max*sample_time
        
        # Clip the normalized progress difference to [-1, 1]
        normalized_linear_progress_reward = np.clip(progress_diff/max_progress_diff, -1, 1)

        return normalized_linear_progress_reward

    def update_reward_buffers(self, component, value):
        self.reward_buffers[component].append(value)

    def render(self):       
        self.render_remove_old_data()
        self.render_add_data()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render_init_plot(self):
        plt.ion()
        
        # Determine the number of subplots based on the reward_plot_map
        num_subplots = max(self.reward_plot_map.values())
        
        self.fig, self.axes = plt.subplots(num_subplots, 1, figsize=(10, 5*num_subplots))
        
        # If there's only one subplot, wrap it in a list for consistency
        if num_subplots == 1:
            self.axes = [self.axes]
        
        self.reward_plots = [{} for _ in range(num_subplots)]
        
        for i, ax in enumerate(self.axes):
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.set_ylim(-15, 15)
            ax.grid()
        
        legend_handles = [
            Line2D([0], [0], **style, label=label)
            for label, style in self.reward_style_map.items()
        ]

        self.fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=self.fig.transFigure)

    def render_add_data(self):
        for component, buffer in self.reward_buffers.items():
            plot_index = self.reward_plot_map.get(component, 0) - 1
            if plot_index >= 0:
                style = self.reward_style_map.get(component, {'color': 'gray', 'linestyle': ':'})
                self.reward_plots[plot_index][component] = self.axes[plot_index].plot(range(len(buffer)), buffer, **style)

    def render_remove_old_data(self):
        for reward_plot in self.reward_plots:
            for plot_list in reward_plot.values():
                for plot in plot_list:
                    plot.remove()