#!/usr/bin/env python3

import utils.admin_tools as at
import utils.base_tools as bt
import os

# Define the base directory path
base_dir = os.path.join(os.path.abspath(os.pardir), 'parameters', 'map_nrs')

# Create and save map_nr_lists
train_map_nr_list, test_map_nr_list = bt.sample_training_test_map_nrs(first_map_idx=0, last_map_idx=299, training_ratio=0.7)
at.save_to_json(train_map_nr_list, 'train_map_nr_list.json', base_dir)
at.save_to_json(test_map_nr_list, 'test_map_nr_list.json', base_dir)

# Create and save map_nr_dicts
train_map_nr_dict = bt.create_level_dictionary(train_map_nr_list, num_levels=5, total_maps=300)
test_map_nr_dict = bt.create_level_dictionary(test_map_nr_list, num_levels=5, total_maps=300)
at.save_to_json(train_map_nr_dict, 'train_map_nr_dict.json', base_dir)
at.save_to_json(test_map_nr_dict, 'test_map_nr_dict.json', base_dir)

# Create and save sampled map_nr_lists
maps_per_level = 5
lowest_level = 1
highest_level = 5
sampled_train_map_nr_list, sampled_test_map_nr_list = bt.sample_maps(train_map_nr_dict, test_map_nr_dict, maps_per_level, lowest_level, highest_level)
at.save_to_json(sampled_train_map_nr_list, 'sampled_train_map_nr_list.json', base_dir)
at.save_to_json(sampled_test_map_nr_list, 'sampled_test_map_nr_list.json', base_dir)
