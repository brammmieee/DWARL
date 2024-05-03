#!/usr/bin/python3

import math
import random
import utils.admin_tools as at

def sample_training_test_map_nrs(range_start, range_end, training_ratio):
    all_numbers = list(range(range_start, range_end + 1))
    print('all nums', all_numbers)

    training_list_size = int(len(all_numbers)*training_ratio)
    print('training_list_size', training_list_size)

    training_list = random.sample(all_numbers, training_list_size)
    testing_list = [x for x in all_numbers if x not in training_list]
    
    return training_list, testing_list

def sample_training_test_map_nrs2(map_list, training_ratio):
    training_list_size = math.floor(len(map_list)*training_ratio)
    print('training_list_size', training_list_size)

    training_list = random.sample(map_list, training_list_size)
    testing_list = [x for x in map_list if x not in training_list]
    
    return training_list, testing_list

# Create the training and test map lists
train_map_nr_list, test_map_nr_list = sample_training_test_map_nrs(range_start=0, range_end=299, training_ratio=0.7)

# Save map lists to pickle files #NOTE: proceed with caution, files will be overwritten
at.write_pickle_file('train_map_nr_list', 'parameters', train_map_nr_list)
at.write_pickle_file('test_map_nr_list', 'parameters', test_map_nr_list)

# Check if everything went well
train_map_nr_list = at.read_pickle_file('train_map_nr_list', 'parameters')
test_map_nr_list = at.read_pickle_file('test_map_nr_list', 'parameters')
print(f'train_map_nr_list = {train_map_nr_list}')
print(f'test_map_nr_list = {test_map_nr_list}')
