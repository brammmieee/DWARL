import os
import datetime
import json
import random
from typing import List, Tuple, Dict

def get_date_time():
    current_date = datetime.datetime.now()  # Returns current date and time
    formatted_date = current_date.strftime("%y_%m_%d")
    formatted_time = current_date.strftime("%H_%M_%S")
    return formatted_date, formatted_time

def find_file(filename, start_dir=os.path.abspath(os.pardir)):
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    
    raise FileNotFoundError(f"File '{filename}' not found starting from directory '{start_dir}'.")

def get_latest_model_dir():
    # Get the latest model directory
    package_dir = os.path.abspath(os.pardir)
    model_dir = os.path.join(package_dir, 'training', 'archive', 'models')
    model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if model_dirs:
        latest_model_dir = max(model_dirs)
        return os.path.join(model_dir, latest_model_dir)
    else:
        raise FileNotFoundError("No model directory found.")

def get_latest_model_steps(model_dir):
    # Get the latest model steps
    model_steps = []
    for d in os.listdir(model_dir):
        if d.endswith('.zip'):
            parts = d.split('_')
            if parts[-1].startswith('steps'):
                step_count = int(parts[-2])
                model_steps.append(step_count)
    
    if model_steps:
        return max(model_steps)
    else:
        return None
    
def sample_training_test_map_nrs(first_map_idx: int, last_map_idx: int, training_ratio: float) -> Tuple[List[int], List[int]]:
    """
    Splits a range of numbers into training and testing lists based on a given ratio.
    
    Args:
    first_map_idx (int): The first index of the map number range.
    last_map_idx (int): The last index of the map number range.
    training_ratio (float): The fraction of the total range to be used for training.
    
    Returns:
    Tuple[List[int], List[int]]: A tuple containing two lists:
        - The first list contains the training indices.
        - The second list contains the testing indices.
    """
    all_numbers = list(range(first_map_idx, last_map_idx + 1))
    training_list_size = int(len(all_numbers) * training_ratio)
    training_list = random.sample(all_numbers, training_list_size)
    testing_list = [x for x in all_numbers if x not in training_list]
    
    return training_list, testing_list

def create_level_dictionary(input_list: List[int], num_levels: int, total_maps: int) -> Dict[str, List[int]]:
    """
    Organizes a list of map indices into a dictionary based on specified levels.
    
    Args:
    input_list (List[int]): The list of integers (map indices) to be organized.
    num_levels (int): The number of levels to divide the maps into.
    total_maps (int): The total number of maps.
    
    Returns:
    Dict[str, List[int]]: A dictionary where each key corresponds to a level ("lvl x") and each value is a list of integers assigned to that level.
    """
    level_dict = {}
    maps_per_level = total_maps // num_levels  # Assumes an equal distribution of maps across levels
    for i in range(1, num_levels + 1):
        lower_bound = (i - 1) * maps_per_level
        if i < num_levels:
            upper_bound = i * maps_per_level
        else:
            upper_bound = total_maps  # Ensure the last level includes any remaining maps due to integer division

        level_key = f'lvl {i}'
        level_values = [num for num in input_list if lower_bound <= num < upper_bound]
        level_dict[level_key] = level_values

    return level_dict

def sample_maps(train_map_nr_dict, test_map_nr_dict, maps_per_level, lowest_level, highest_level):
    """
    Sample maps from level-based dictionaries for training and testing.

    Args:
    train_map_nr_dict (dict): A dictionary containing training maps categorized by levels.
    test_map_nr_dict (dict): A dictionary containing testing maps categorized by levels.
    maps_per_level (int): Number of maps to sample per level.
    lowest_level (int): Lowest level to consider.
    highest_level (int): Highest level to consider.

    Returns:
    tuple: Two lists containing the sampled training and testing map numbers.
    """
    train_map_nr_list = []
    test_map_nr_list = []
    
    for i, map_nr_dict in enumerate([train_map_nr_dict, test_map_nr_dict]):
        for lvl in range(lowest_level, highest_level + 1):
            level_key = f'lvl {lvl}'
            map_nr_list = map_nr_dict.get(level_key, [])
            if i == 0:  # Training maps
                train_map_nr_list.extend(sorted(random.sample(map_nr_list, maps_per_level)))
            else:  # Testing maps
                test_map_nr_list.extend(sorted(random.sample(map_nr_list, maps_per_level)))
                
    return train_map_nr_list, test_map_nr_list