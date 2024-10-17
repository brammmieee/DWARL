#!/usr/bin/env python3

import utils.admin_tools as at
import os

# Define the base directory path
base_dir = os.path.join(os.path.abspath(os.pardir), 'parameters', 'map_nrs')

# Create and display map_nr_lists
train_map_nr_list, test_map_nr_list = at.sample_training_test_map_nrs(first_map_idx=0, last_map_idx=299, training_ratio=0.7)
print("Train Map Number List:")
print(train_map_nr_list)
print("\nTest Map Number List:")
print(test_map_nr_list)

# Create and display map_nr_dicts
train_map_nr_dict = at.create_level_dictionary(train_map_nr_list, num_levels=5, total_maps=300)
test_map_nr_dict = at.create_level_dictionary(test_map_nr_list, num_levels=5, total_maps=300)
print("\nTrain Map Number Dictionary:")
print(train_map_nr_dict)
print("\nTest Map Number Dictionary:")
print(test_map_nr_dict)

# Create and display sampled map_nr_lists
maps_per_level = 5
lowest_level = 1
highest_level = 5
sampled_train_map_nr_list, sampled_test_map_nr_list = at.sample_maps(train_map_nr_dict, test_map_nr_dict, maps_per_level, lowest_level, highest_level)
print("\nSampled Train Map Number List:")
print(sampled_train_map_nr_list)
print("\nSampled Test Map Number List:")
print(sampled_test_map_nr_list)
