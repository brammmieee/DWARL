# %%
%load_ext autoreload
%autoreload # reloads custom modules into Ipython interpreter
%matplotlib qt

# %%
import os 
import matplotlib.pyplot as plt
import numpy as np
from utils import custom_tools as ct
from environments.custom_test_env import CustomTestEnv
from stable_baselines3.ppo import PPO
import random

def compute_list_average(list):
    total_sum = sum(list)
    num_elements = len(list)
    if num_elements == 0:
        return 0  # Handle the case when the list is empty to avoid division by zero
    else:
        average = total_sum / num_elements
        return average

# %%
np.set_printoptions(precision=5, suppress=True)
package_dir = os.path.abspath(os.pardir)
model_archive_dir = os.path.join(package_dir, 'testing', 'models', 'archived')

# ============================== # Creating Env + Loading Model # ======================== #
# %% Standard env
env = CustomTestEnv(render_mode=None, wb_open=True, wb_mode='training')

# %% Loading model
model_name = 'insert model name'
n_steps_load = 00000 
model = PPO.load('models' + '/' + model_name + '/' + model_name + '_' + str(n_steps_load) + '_steps')

# =================================== # Running Tests # ================================== #
# %% Testing variables
map_nr_list = ct.read_pickle_file('train_map_nr_list', os.path.join('testing', 'map numbers'))
tests_per_map = 1       # how many itterations to run per map
time_limit = 30         # terminates test when exceeded [s]
goal_tolerance = 0.3   # [m]

# %% Running tests
avg_traversal_time_list = []
avg_succes_rate_list = []
done_cause_list = ['timed_out', 'arrived_at_goal', 'got_stuck', 'out_of_bound']

for map_nr in map_nr_list:
    print(f'Testing on map number {map_nr}')
    options = {
        'map_nr': map_nr, 
        'max_ep_time': time_limit, 
        'goal_tolerance': goal_tolerance, 
        'plot_trajectories': False, 
        'step_size': 25
    }
    test_nr = 1
    succes_list = []
    traversal_time_list = []
    observation = env.reset(options)

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, done, info = env.step(action)
        
        traversal_time = info['sim_time'] #[s]
        done_cause = info['done_cause']

        if done:
            if done_cause == 'arrived_at_goal':
                succes = 1
                print(f'Test {test_nr} succesfully completed -> traversal time = {traversal_time}')
            else: 
                succes = 0
                print(f'Test {test_nr} failed -> done cause = {done_cause}')
            
            succes_list.append(succes)
            traversal_time_list.append(traversal_time)

            if test_nr == tests_per_map:
                break 

            observation = env.reset(options)
            test_nr += 1
    
    avg_traversal_time_list.append(compute_list_average(traversal_time_list))
    avg_succes_rate_list.append(compute_list_average(succes_list))

env.close()
print('Testing complete')

ct.write_pickle_file('avg_traversal_time_list', os.path.join('testing','results'), avg_traversal_time_list)
ct.write_pickle_file('avg_succes_rate_list', os.path.join('testing','results'), avg_succes_rate_list)

count = 0
for item in avg_succes_rate_list:
    if item == 1:
        count +=1

print(f'{count} out of 25 succes')











# =========================== # Creating testing map dicts # ============================== #
# %% Loading training and testing map nr lists
train_map_nr_list = ct.read_pickle_file('train_map_nr_list', 'parameters')
test_map_nr_list = ct.read_pickle_file('test_map_nr_list', 'parameters')

# %% Creating leveled dicts
def create_level_dictionary(input_list):
    level_dict = {}
    
    for i in range(1, 6):
        lower_bound = (i - 1) * 60
        upper_bound = i * 60
        level_key = f'lvl {i}'
        level_values = [num for num in input_list if lower_bound <= num < upper_bound]
        level_dict[level_key] = level_values

    return level_dict

train_map_nr_dict = create_level_dictionary(train_map_nr_list)
test_map_nr_dict  = create_level_dictionary(test_map_nr_list)

# %% Writing to file (NOTE: be cautious, overwrites old ones)
# ct.write_pickle_file('train_map_nr_dict', os.path.join('testing', 'map numbers'), train_map_nr_dict)
# ct.write_pickle_file('test_map_nr_dict', os.path.join('testing', 'map numbers'), test_map_nr_dict)






# ========================= # Creating map nr list from level dicts # ==================== #
# %%
maps_per_level = 5
train_map_nr_dict = ct.read_pickle_file('train_map_nr_dict', os.path.join('testing', 'map numbers'))
test_map_nr_dict = ct.read_pickle_file('test_map_nr_dict', os.path.join('testing', 'map numbers'))

train_map_nr_list = []
test_map_nr_list = []
for i, map_nr_dict in enumerate([train_map_nr_dict], [test_map_nr_dict]):
    for lvl in range(1,6):
        level_key = f'lvl {lvl}'
        map_nr_list = map_nr_dict[level_key]
        if i == 0: # training maps
            for map_nr in sorted(random.sample(map_nr_list, maps_per_level)):
                train_map_nr_list.append(map_nr) 
        elif i == 1: # testing maps
            for map_nr in sorted(random.sample(map_nr_list, maps_per_level)):
                test_map_nr_list.append(map_nr) 

# %% Writing to file (NOTE: be cautious, overwrites old ones)
# ct.write_pickle_file('train_map_nr_list', os.path.join('testing', 'map numbers'), train_map_nr_list)
# ct.write_pickle_file('test_map_nr_list', os.path.join('testing', 'map numbers'), test_map_nr_list)