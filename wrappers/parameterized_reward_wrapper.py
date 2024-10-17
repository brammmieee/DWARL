import numpy as np
import gymnasium as gym

class ParameterizedRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Reward buffer and plotting
        reward_plotting_config = at.load_parameters(['reward_plotting_config.json'])
        self.reward_plot_map = {key: value["plot_nr"] for key, value in reward_plotting_config.items()}

        self.reward_buffers = {component: [] for component in reward_plotting_config.keys()}
        self.reward_style_map = {key: value["plot_style"] for key, value in reward_plotting_config.items() if self.reward_plot_map.get(key, 0) != 0}
        
    def step():
        # called after update_robot_state_and_local_goal in baseEnv
        self.update_path_dist_progress_and_heading()

    def reset():
        self.reset_path_dist_progress_and_heading()
    
    def reward(self, reward):
        modified_reward = self.get_reward(self.unwrapped.done, self.unwrapped.done_cause)
        return modified_reward
    
    def update_reward_buffers(self, reward_components, total_reward):
        for key, value in reward_components.items():
            self.reward_buffers[key].append(value)
            if len(self.reward_buffers[key]) > self.params['reward_buffer_size']:
                self.reward_buffers[key].pop(0)
                
        self.reward_buffers['total_reward'].append(total_reward)
        if len(self.reward_buffers['total_reward']) > self.params['reward_buffer_size']:
            self.reward_buffers['total_reward'].pop(0)

    def reset_path_dist_progress_and_heading(self):
        self.path_dist = 0
        self.prev_path_dist = 0

        self.path_progress = 0
        self.prev_path_progress = 0

        self.path_heading = 0
        self.prev_path_heading = 0

        self.init_progress = self.calculate_progress(self.init_pose)
        self.goal_progress = self.calculate_progress(self.goal_pose)

    def update_path_dist_progress_and_heading(self):
        path_dist = np.inf
        path_progress = 0
        path_heading = 0
        cumulative_path_length = 0
        r = self.cur_pos[:2]
        psi = (np.arctan2(self.cur_orient_matrix[3], self.cur_orient_matrix[0])) % (2*np.pi)

        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i + 1])
            v = p2 - p1
            w = r - p1
            t = np.dot(w, v) / np.dot(v, v)
            t = max(0, min(1, t))

            closest_point_on_segment = p1 + t * v
            segment_dist = np.linalg.norm(r - closest_point_on_segment)

            # Caculation of path_dist, path_progress, and path_heading
            if segment_dist < path_dist:
                # Path dist calculation
                path_dist = segment_dist

                # Path progress calculation
                segment_progress = cumulative_path_length + t * np.linalg.norm(v)
                if self.direction > 0:
                    path_progress = segment_progress - self.init_progress
                else:
                    path_progress = self.init_progress - segment_progress
                
                # Path heading calculation
                path_angle = np.arctan2(v[1], v[0])
                if self.direction < 0:
                    path_angle += np.pi
                path_angle = path_angle % (2*np.pi)

                path_heading = np.abs(psi - path_angle)

            cumulative_path_length += np.linalg.norm(v)

        # Adjust negative progress condition based on direction
        if self.direction > 0 and path_progress > self.goal_progress:
            path_progress = -(path_progress - self.goal_progress)
        elif self.direction < 0 and path_progress < self.goal_progress:
            path_progress = -(self.goal_progress - path_progress)

        self.prev_path_dist = self.path_dist
        self.path_dist = path_dist if path_dist < np.inf else 0
        self.prev_path_progress = self.path_progress
        self.path_progress = path_progress
        self.prev_path_heading = self.path_heading
        self.path_heading = path_heading

    def calculate_progress(self, pose):
        total_length = 0
        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i])
            p2 = np.array(self.path[i + 1])
            if np.array_equal(p1, pose[:2]):
                progress = total_length
                break
            if np.array_equal(p2, pose[:2]):
                progress = total_length + np.linalg.norm(p2 - p1)
                break
            total_length += np.linalg.norm(p2 - p1)

        return progress

    def get_reward(self, done, done_cause):    
        # Initialize reward components dictionary with empty values
        reward_components = {
            'r_at_goal': 0,
            'r_outside_map': 0,
            'r_collision': 0,
            'r_not_arrived': 0,
            'r_path_d_dist': 0,
            'r_path_d_progress': 0,
            'r_linear_d_progress': 0,
            'r_path_d_heading': 0,
            'r_path_dist': 0,
            'r_path_heading': 0,
            'r_path_d_progress_scaled_dist': 0,
        }

        if done:
            # Assign reward based on the done cause
            if done_cause == 'at_goal':
                reward_components['r_at_goal'] = self.params['r_at_goal']
            elif done_cause == 'outside_map':
                reward_components['r_outside_map'] = self.params['r_outside_map']
            elif done_cause == 'collision':
                reward_components['r_collision'] = self.params['r_collision']
            else:
                raise ValueError(f'get_reward(): done_cause "{done_cause}" not recognized')
        else:
            # Calculate ongoing rewards
            reward_components['r_not_arrived'] = self.params['r_not_arrived']

            reward_components['r_path_d_dist'] = self.params['c_path_d_dist']*self.get_normalized_path_d_dist()
            normalized_path_progress = self.get_normalized_path_d_progress()
            reward_components['r_path_d_progress'] = self.params['c_path_d_progress']*normalized_path_progress
            reward_components['r_linear_d_progress'] = self.params['c_linear_d_progress']*self.get_normalized_linear_d_progress()
            reward_components['r_path_d_heading'] = self.params['c_path_d_heading']*self.get_normalized_path_d_heading()
            
            normalized_path_dist_reward = self.get_normalized_path_dist_reward()
            reward_components['r_path_dist'] = self.params['c_path_dist']*normalized_path_dist_reward
            reward_components['r_path_heading'] = self.params['c_path_heading']*self.get_normalized_path_heading_reward()
            reward_components['r_path_d_progress_scaled_dist'] = self.params['c_path_d_progress_scaled_dist']*self.get_normalized_d_pogress_scaled_path_dist_reward(normalized_path_progress, normalized_path_dist_reward)
            
        # Calculate total reward as the sum of all components
        total_reward = sum(reward_components.values())

        # Append rewards to reward buffers for plotting
        if self.render_mode is not None:
            self.update_reward_buffers(reward_components, total_reward)

        return total_reward
        
    def get_normalized_path_d_dist(self):
        if self.params['c_path_d_dist'] == 0:
            return 0    
        path_dist_diff = -(self.path_dist - self.prev_path_dist)
        max_path_dist = self.params['v_max']*self.params['sample_time']
        normalized_path_dist = np.clip(path_dist_diff / max_path_dist, -1, 1)

        return normalized_path_dist
    
    def get_normalized_path_d_progress(self):
        if self.params['c_path_d_progress'] == 0 and self.params['c_path_d_progress_scaled_dist'] == 0:
            return 0
        path_progress_diff = self.path_progress - self.prev_path_progress
        max_path_progress = self.params['v_max']*self.params['sample_time']
        
        if not self.params['enable_d_progress_mapping']:
            normalized_path_progress_diff = path_progress_diff / max_path_progress
        else:
            normalized_path_progress_diff = self.get_mapped_normalized_progress_diff(path_progress_diff)

        normalized_path_progress = np.clip(normalized_path_progress_diff, -1, 1)

        return normalized_path_progress
    
    def get_normalized_linear_d_progress(self):
        if self.params['c_linear_d_progress'] == 0:
            return 0
        prev_dist_to_goal = np.linalg.norm(self.goal_pose[:2] - self.prev_pos[:2])
        current_dist_to_goal = np.linalg.norm(self.goal_pose[:2] - self.cur_pos[:2])
        progress_diff = prev_dist_to_goal - current_dist_to_goal
        max_progress_diff = self.params['v_max']*self.params['sample_time']
        normalized_progress_diff = np.clip(progress_diff / max_progress_diff, -1, 1)
        
        return normalized_progress_diff

    def get_mapped_normalized_progress_diff(self, path_progress_diff):
        # print(f"x: {path_progress_diff}")
        if path_progress_diff < self.params['d_progress_low_pass_x']:
            progress_diff = -1
        elif path_progress_diff > self.params['d_progress_high_pass_x']:
            progress_diff = 1
        else:
            x = path_progress_diff
            progress_diff = -674663.913519*x**3 + 5658.877973*x**2 + 213.837054*x**1 - 1.029126
        # print(f"f(x): {progress_diff}")
        return progress_diff
            
    def get_normalized_path_d_heading(self):
        if self.params['c_path_d_heading'] == 0:
            return 0
        path_heading_diff = -(self.path_heading - self.prev_path_heading)
        max_path_heading = self.params['omega_max']*self.params['sample_time']
        normalized_path_heading = np.clip(path_heading_diff / max_path_heading, -1, 1)

        return normalized_path_heading
    
    def get_normalized_path_dist_reward(self):
        if self.params['c_path_dist'] == 0 and self.params['c_path_d_progress_scaled_dist'] == 0:
            return 0    
        max_distance = self.params['max_path_dist']
        normalized_path_dist = self.path_dist / max_distance
        normalized_path_dist_reward = np.clip(1 - normalized_path_dist, -1, 1)
        
        return normalized_path_dist_reward

    def get_normalized_path_heading_reward(self):
        if self.params['c_path_heading'] == 0:
            return 0
        max_path_heading = self.params['max_path_heading']
        normalized_path_heading = self.path_heading / max_path_heading
        normalized_path_heading_reward = np.clip(1 - normalized_path_heading, -1, 1)

        return normalized_path_heading_reward
    
    def get_normalized_d_pogress_scaled_path_dist_reward(self, normalized_path_progress, normalized_path_dist_reward):
        if self.params['c_path_d_progress_scaled_dist'] == 0:
            return 0
        
        if np.clip(normalized_path_progress, 0, 1) >= 0.0:
            return np.clip(normalized_path_progress, 0, 1)*normalized_path_dist_reward
        else:
            return normalized_path_dist_reward
        
# INIT STUFF

# # ax3 - Reward plot
# self.ax3.set_xlabel('Step')
# self.ax3.set_ylabel('Reward')
# self.ax3.set_ylim(-15, 15)
# self.ax3.grid()
# self.reward_plots_1 = {}

# # ax4 - dReward/dStep plot
# self.ax4.set_xlabel('Step')
# self.ax4.set_ylabel('Reward')
# self.ax4.set_ylim(-15, 15)
# self.ax4.grid()
# self.reward_plots_2 = {}

# # REMOVE STUFF

# # ax3 - Clear reward plot
# try:
#     for reward_plot in self.reward_plots_1.values():
#         for plot in reward_plot:
#             plot.remove()
# except AttributeError:
#     pass

# # ax4 - Clear dReward/dStep plot
# try:
#     for reward_plot in self.reward_plots_2.values():
#         for plot in reward_plot:
#             plot.remove()
# except AttributeError:
#     pass


# # ax3 - Reward plot
# for component, buffer in self.reward_buffers.items():
#     if self.reward_plot_map[component] != 1:
#         continue
#     style = self.reward_style_map.get(component, {'color': 'gray', 'linestyle': ':'})  # Default style
#     self.reward_plots_1[component] = self.ax3.plot(range(len(buffer)), buffer, label=component, **style)

# # ax4 - dReward/dStep plot
# for component, buffer in self.reward_buffers.items():
#     if self.reward_plot_map[component] != 2:
#         continue
#     style = self.reward_style_map.get(component, {'color': 'gray', 'linestyle': ':'})  # Default style
#     self.reward_plots_2[component] = self.ax4.plot(range(len(buffer)), buffer, label=component, **style)

