import os
import json
import shutil
import numpy as np

def mkdir_if_not_exists(dir_name: str, recursive: bool=False) -> None:
    """ Create a directory with the specified name if it does not already exist.
    Supports both single-level and recursive directory creation.

    Args:
        dir_name [str]: The name or path of the directory to create.
        recursive [bool]: Whether to create directories recursively (including parent directories).
        
    Return:
        None. The function creates the directory if it does not exist.
    """
    if os.path.exists(dir_name):
        return 
    
    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)

def rmdir_if_exists(dir_name: str, recursive: bool=False) -> None:
    """ Remove the specified directory if it exists.

    Args:
        dir_name [str]: The name or path of the directory to remove.
        recursive [bool]: Whether to remove directories recursively (including all contents).

    Return:
        None. The function removes the directory if it exists.
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

class NumpyArrayEncoder(json.JSONEncoder):
    """ Python dictionaries can store ndarray array types, but when serialized by dict into JSON files, 
    ndarray types cannot be serialized. In order to read and write numpy arrays, 
    we need to rewrite JSONEncoder's default method. 
    The basic principle is to convert ndarray to list, and then read from list to ndarray.
    """
    def default(self, obj):
        return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dict2json(file_name: str, the_dict: dict) -> None:
    """ Save a Python dictionary object as a JSON file. The function serializes the dictionary,
    including support for NumPy arrays via a custom encoder, and writes it to the specified file.
    Returns 1 on success, 0 on failure.

    Args:
        file_name [str]: The path and name of the file where the JSON data will be saved.
        the_dict [dict]: The dictionary object to be serialized and saved as JSON.

    Return:
        int: Returns 1 if the operation is successful, otherwise returns 0.
    """
    try:
        json_str = json.dumps(the_dict, cls=NumpyArrayEncoder, indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        return 0 