#!/usr/bin/python3

from utils import custom_tools as ct
import os

# Loading training and testing map nr lists
train_map_nr_list = ct.read_pickle_file('train_map_nr_list', 'parameters')
test_map_nr_list = ct.read_pickle_file('test_map_nr_list', 'parameters')

# Creating leveled dicts
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

# Writing to file (NOTE: be cautious, overwrites old ones)
ct.write_pickle_file('train_map_nr_dict', os.path.join('testing', 'map numbers'), train_map_nr_dict)
ct.write_pickle_file('test_map_nr_dict', os.path.join('testing', 'map numbers'), test_map_nr_dict)



