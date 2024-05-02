import os
import pickle
import datetime

def write_pickle_file(file_name, file_dir, value):
    # NOTE: file dir is relative to package folder
    package_dir = os.path.abspath(os.pardir)
    pickle_file_path = os.path.join(package_dir, file_dir, file_name + '.pickle')
    with open(pickle_file_path, "wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    
def read_pickle_file(file_name, file_dir):
    package_dir = os.path.abspath(os.pardir)
    pickle_file_path = os.path.join(package_dir, file_dir, file_name + '.pickle')
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