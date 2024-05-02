#!/usr/bin/python3

from utils import custom_tools as ct
import os
import random

# Creating map nr list from level dicts
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

# Writing to file (NOTE: be cautious, overwrites old ones)
ct.write_pickle_file('train_map_nr_list', os.path.join('testing', 'map numbers'), train_map_nr_list)
ct.write_pickle_file('test_map_nr_list', os.path.join('testing', 'map numbers'), test_map_nr_list)