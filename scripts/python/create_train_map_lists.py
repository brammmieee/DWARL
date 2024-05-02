#!/usr/bin/python3

import utils.custom_tools as ct

# Create the training and test map lists
train_map_nr_list, test_map_nr_list = ct.sample_training_test_map_nrs(range_start=0, range_end=299, training_ratio=0.7)

# Save map lists to pickle files #NOTE: proceed with caution, files will be overwritten
ct.write_pickle_file('train_map_nr_list', 'parameters', train_map_nr_list)
ct.write_pickle_file('test_map_nr_list', 'parameters', test_map_nr_list)

# Check if everything went well
train_map_nr_list = ct.read_pickle_file('train_map_nr_list', 'parameters')
test_map_nr_list = ct.read_pickle_file('test_map_nr_list', 'parameters')
print(f'train_map_nr_list = {train_map_nr_list}')
print(f'test_map_nr_list = {test_map_nr_list}')
