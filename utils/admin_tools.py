import os
import yaml
import json
import pickle
import datetime
from typing import List, Union, Any, Dict

def write_pickle_file(file_name, file_dir, value):
    # NOTE: file dir is relative to package folder
    package_dir = os.path.abspath(os.pardir)
    pickle_file_path = os.path.join(package_dir, file_dir, file_name + '.pickle')
    with open(pickle_file_path, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    
def read_pickle_file(file_name, file_dir):
    pickle_file_path = os.path.join(file_dir, file_name + '.pickle')
    with open(pickle_file_path, "rb") as file:
        pickle_file = pickle.load(file)
        return pickle_file

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
        return None

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

def load_parameters(file_name_list: Union[str, List[str]], start_dir=os.path.abspath(os.pardir)):
    """
    Load parameters from YAML or JSON files based on the given file names, searching from a specific start directory or default directory.

    Args:
        file_name_list (Union[str, List[str]]): A single file name or a list of file names.
        start_dir (str, optional): The directory from which to start the search for files.

    Returns:
        dict: A dictionary containing the loaded parameters.

    """
    if isinstance(file_name_list, str):
        file_name_list = [file_name_list]

    parameters = {}
    for file_name in file_name_list:
        try:
            file_path = find_file(file_name, start_dir)
            _, file_extension = os.path.splitext(file_path)
            with open(file_path, 'r') as stream:
                if file_extension == '.yaml' or file_extension == '.yml':
                    try:
                        parameters.update(yaml.safe_load(stream))
                    except yaml.YAMLError as exc:
                        print(f"Error loading YAML from {file_path}: {exc}")
                elif file_extension == '.json':
                    try:
                        json_dict = json.load(stream)
                        parameters.update(json_dict)
                    except json.JSONDecodeError as exc:
                        print(f"Error loading JSON from {file_path}: {exc}")
                else:
                    print(f"Unsupported file format for {file_path}")
        except FileNotFoundError as e:
            print(e)

    return parameters

def save_to_json(data: Dict[str, Any], filename: str, file_dir: str) -> None:
    """
    Save data to a JSON file, creating the directory if it doesn't exist.

    Args:
    data (dict): The data to be saved.
    filename (str): The name of the file to save the data in.
    file_dir (str): The directory relative to the package where the file should be saved.
    """
    # Get the directory relative to the package folder
    package_dir = os.path.abspath(os.pardir)
    json_file_path = os.path.join(package_dir, file_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(json_file_path, exist_ok=True)
    
    # Create the full path to the file
    full_path = os.path.join(json_file_path, filename)
    
    # Save the data to JSON file
    with open(full_path, 'w') as json_file:
        json.dump(data, json_file)

def load_from_json(filename: str, file_dir: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
    filename (str): The name of the file to load the data from.
    file_dir (str): The directory relative to the package where the file is located.

    Returns:
    dict: The data loaded from the JSON file.
    """
    # Get the directory relative to the package folder
    package_dir = os.path.abspath(os.pardir)
    json_file_path = os.path.join(package_dir, file_dir, filename)
    
    # Load the data from JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    else:
        raise FileNotFoundError(f"No file found at {json_file_path}")