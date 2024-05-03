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
    
def create_day_folder(file_dir):
    # NOTE: file_dir is the dir relative to the 
    package_dir = os.path.abspath(os.pardir)
    current_date = datetime.datetime.now()
    day_folder = current_date.strftime('%d')  # Format: DD
    directory = os.path.join(package_dir, file_dir, day_folder)

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_name_with_date(test_nr_today, comment=''):
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%B%d")
    return f'{formatted_date}V{str(test_nr_today)}{comment}'

def get_file_name_with_date_testing(test_nr_today, comment):
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%B_%d")
    return f'{formatted_date}_V{str(test_nr_today)}_{comment}'

def load_parameters(file_name_list: Union[str, List[str]]):
    if isinstance(file_name_list, str):
        file_name_list = [file_name_list]

    parameters = {}
    for root, dirs, files in os.walk(os.path.join(os.path.abspath(os.pardir), "parameters")):
        for file in files:
            if file.endswith(".yaml") and any(name in file for name in file_name_list):
                with open(os.path.join(root, file), 'r') as stream:
                    try:
                        parameters.update(yaml.safe_load(stream))
                    except yaml.YAMLError as exc:
                        print(exc)
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