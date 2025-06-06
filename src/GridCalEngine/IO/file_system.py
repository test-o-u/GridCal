# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import os
import importlib
import importlib.util
import hashlib
from pathlib import Path


def get_create_gridcal_folder() -> str:
    """
    Get the home folder of gridCal, and if it does not exist, create it
    :return: folder path string
    """
    home = str(Path.home())

    gc_folder = os.path.join(home, '.GridCal')

    if not os.path.exists(gc_folder):
        os.makedirs(gc_folder)

    return gc_folder


def get_create_dynamics_folder() -> str:
    """
    Generate dynamics folder
    :return: ./GridCal/dynamics
    """
    base_folder = get_create_gridcal_folder()

    dyn_folder = os.path.join(base_folder, "dynamics")

    if not os.path.exists(dyn_folder):
        os.makedirs(dyn_folder)

    return dyn_folder


def get_create_dynamics_model_folder(grid_id: str) -> str:
    """
    Generate dynamics model folder
    :param grid_id: idtag of a MultiCircuit
    :return: ./GridCal/dynamics/model_id
    """
    base_folder = get_create_dynamics_folder()

    dyn_folder = os.path.join(base_folder, grid_id)

    if not os.path.exists(dyn_folder):
        os.makedirs(dyn_folder)

    return dyn_folder


def clear_dynamics_model_folder(grid_id: str):
    """
    Delete all files in the model dynamics folder
    :param grid_id: idtag of a MultiCircuit
    """
    folder_path = get_create_dynamics_model_folder(grid_id=grid_id)

    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)  # Delete the file

        for directory in dirs:
            dir_path = os.path.join(root, directory)
            os.rmdir(dir_path)  # Remove empty directories


def opf_file_path() -> str:
    """
    get the OPF files folder path
    :return: str
    """
    d = os.path.join(get_create_gridcal_folder(), 'mip_files')

    if not os.path.exists(d):
        os.makedirs(d)
    return d


def plugins_path() -> str:
    """
    get the plugins file path
    :return: plugins file path
    """
    pth = os.path.join(get_create_gridcal_folder(), 'plugins')

    if not os.path.exists(pth):
        os.makedirs(pth)

    return pth


def tiles_path() -> str:
    """
    get the tiles file path
    :return: tiles file path
    """
    pth = os.path.join(get_create_gridcal_folder(), 'tiles')

    if not os.path.exists(pth):
        os.makedirs(pth)

    return pth


def scripts_path() -> str:
    """
    get the scripts file path
    :return: scripts file path
    """
    pth = os.path.join(get_create_gridcal_folder(), 'scripts')

    if not os.path.exists(pth):
        os.makedirs(pth)

    return pth


def api_keys_path() -> str:
    """
    get the api keys file path
    :return: api keys file path
    """
    pth = os.path.join(get_create_gridcal_folder(), 'api_keys')

    if not os.path.exists(pth):
        os.makedirs(pth)

    return pth


def load_file_as_module(file_path: str):
    """
    Dynamically load a function from a Python file at a given file path.

    :param file_path: The path to the Python (.py) file.
    :return: The loaded function object.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ImportError: If the module cannot be imported.
    :raises AttributeError: If the function does not exist in the module.
    :raises TypeError: If the retrieved attribute is not callable.
    """
    # Ensure the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    # Generate a unique module name to avoid conflicts
    # Here, we use the file's absolute path hashed to ensure uniqueness
    absolute_path = os.path.abspath(file_path)
    module_name = f"dynamic_module_{hashlib.md5(absolute_path.encode()).hexdigest()}"

    # Create a module specification from the file location
    spec = importlib.util.spec_from_file_location(module_name, absolute_path)
    if spec is None:
        raise ImportError(f"Cannot create a module spec for '{file_path}'")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    try:
        # Execute the module to populate its namespace
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to execute module '{file_path}': {e}") from e


    return module


def load_function_from_module(module, function_name: str):
    """
    Dynamically load a function from a Python file at a given file path.
    :param module: some python module
    :param function_name: The name of the function to load from the file.
    :return: The loaded function object.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ImportError: If the module cannot be imported.
    :raises AttributeError: If the function does not exist in the module.
    :raises TypeError: If the retrieved attribute is not callable.
    """

    # Retrieve the function from the module
    if not hasattr(module, function_name):
        raise AttributeError(f"The function '{function_name}' does not exist in module")

    func = getattr(module, function_name)

    if not callable(func):
        raise TypeError(f"'{function_name}' in module is not callable")

    return func

def load_var_from_module(module, var_name: str):
    """
    Dynamically load a function from a Python file at a given file path.
    :param module: some python module
    :param var_name: The name of the variable to load from the file.
    :return: The loaded function object.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ImportError: If the module cannot be imported.
    :raises AttributeError: If the function does not exist in the module.
    :raises TypeError: If the retrieved attribute is not callable.
    """

    # Retrieve the function from the module
    if not hasattr(module, var_name):
        raise AttributeError(f"The variable '{var_name}' does not exist in module")

    func = getattr(module, var_name)

    return func


def load_function_from_file_path(file_path: str, function_name: str):
    """
    Dynamically load a function from a Python file at a given file path.

    :param file_path: The path to the Python (.py) file.
    :param function_name: The name of the function to load from the file.
    :return: The loaded function object.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises ImportError: If the module cannot be imported.
    :raises AttributeError: If the function does not exist in the module.
    :raises TypeError: If the retrieved attribute is not callable.
    """
    module = load_file_as_module(file_path=file_path)

    # Retrieve the function from the module
    if not hasattr(module, function_name):
        raise AttributeError(f"The function '{function_name}' does not exist in '{file_path}'")

    func = getattr(module, function_name)

    if not callable(func):
        raise TypeError(f"'{function_name}' in '{file_path}' is not callable")

    return func