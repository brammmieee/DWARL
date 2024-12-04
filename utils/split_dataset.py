# %%  # Script for semi randomly creating test and train sets for barns
import numpy as np
from pathlib import Path
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt

# %%
def get_intervals(max_nr_maps=300, ratio=0.8):
    
    nr_test_maps = int(np.round(max_nr_maps * (1-ratio)))
    average = max_nr_maps / nr_test_maps
    ce = int(np.ceil(average))
    fl = int(np.floor(average))
    if ce == fl:
        sample_list = nr_test_maps * [ce]
    else:
        nr_floors = int(np.round(nr_test_maps*(-average + ce)/(ce - fl)))
        nr_ceils = int(np.round(nr_test_maps*(average - fl)/(ce - fl)))
        sample_list = nr_floors * [fl] + nr_ceils * [ce]

    return sample_list

def get_semi_random_map_nrs(max_nr_maps=300, ratio=0.8):
    intervals = get_intervals(max_nr_maps, ratio)
    np.random.shuffle(intervals)
    i = 0
    test_nrs = []
    train_nrs = []
    for s in intervals:
        test_nr = np.random.randint(i, i+s)
        test_nrs.append(test_nr)
        train_nr = [n for n in range(i, i+s) if n != test_nr]
        train_nrs = train_nrs + train_nr
        i += s
    return test_nrs, train_nrs

def print_barn_list(nrs):
    for i in nrs:
        print(f'- barn_{i}')

# %%
ratio = 0.8
max_nr_maps = 300
test_nrs, train_nrs = get_semi_random_map_nrs(ratio=0.8)
test_nrs, train_nrs
all_nrs = test_nrs + train_nrs
unordered_list = list(map(int, np.arange(max_nr_maps)))
assert sorted(all_nrs) == sorted(unordered_list), "The numbers in all_nrs do not match the unordered_list"

# %% Manually save these to yaml:
print_barn_list(test_nrs)
print_barn_list(train_nrs)

