# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import importlib
import compileall
import time  
import pdb
import numpy as np
import sympy as sp
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Devices.Dynamic.dae import DAE
from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path
from GridCalEngine.Devices.Dynamic.io.json import readjson


class System:
    """
    This class represents a power system containing various models and devices.

    It handles:
    - Importing and managing abstract dynamic models.
    - Processing symbolic and numerical computations.
    - Creating and managing device instances in one single object in a vecotrized form.
    - Assigning global indices to system variables and copies of these to external system variables.
    """

    def __init__(self, models_list, datafile):
        """
        Initializes the System instance.

        Args:
            models_list (list): A list of model categories and their associated models.
            datafile (str): Path to the JSON file containing device data and system configuration.

        Attributes:
            models_list (list): Stores the provided list of model categories and models.
            models (dict): A dictionary mapping model names to their respective instances.
            devices (dict): A dictionary to store instantiated device objects.
            dae (DAE): An instance of the DAE class for managing algebraic and differential equations.
            data (dict): Parsed JSON data containing device configurations.
        """

        self.models_list = models_list
        self.values_array = None

        self.models = {}
        self.devices = {}

        self.dae = DAE(self)

        self.data = readjson(datafile)

        self.import_models()
        self.system_prepare()


    def import_models(self):
        """
        Imports dynamic models and initializes their symbolic-numeric representations.

        This method:
        - Dynamically imports model classes from GridCalEngine.
        - Creates empty instances of each model and stores them in `self.models`.
        """

        for category, model_names in self.models_list:
            # Import the module containing the models for this category (__init__.py)
            category_module = importlib.import_module(f'GridCalEngine.Devices.Dynamic.models.{category}')
            
            for model_name in model_names:
                # Retrieve the model class from the module
                ModelClass = getattr(category_module, model_name)
                
                # Instantiate the model with default attributes
                model = ModelClass(name=model_name, code='', idtag='')
                
                # Store the model instance in the dictionary
                self.models[model_name] = model
        
    def system_prepare(self):
        """
        Prepares the system by processing models, creating devices, and assigning global indices.

        This method consists of three main steps:
        1. Processing models symbolically and generating optimized numerical functions.
        2. Creating instances of devices and storing them in vectorized form.
        3. Assigning global indices to algebraic variables and external references.

        Execution time for each step is measured and stored.
        """

        # STEP 1: Process models symbolically and generate numerical functions
        symb_st = time.perf_counter()
        for model in self.models.values():
            model.store_data()
            model.process_symbolic()
        
        # Finalize generated code
        self.finalize_pycode()
        symb_end = time.perf_counter()
        self.symb_time = symb_end - symb_st  # Store symbolic processing time

        # STEP 2: Create vectorized model instances for device storage
        dev_st = time.perf_counter()
        self.create_devices(self.data)
        dev_end = time.perf_counter()
        self.dev_time = dev_end - dev_st  # Store device creation time

        # STEP 3: Assign global indices to variables and external references
        add_st = time.perf_counter()
        self.set_addresses()
        add_end = time.perf_counter()
        self.add_time = add_end - add_st  # Store addressing time

        # STEP 4: Store parameters in teh DAE
        self.store_params()

    def create_devices(self, data):
        """
        Populates vectorized model instances with device data from a parsed JSON file.

        This method:
        - Iterates through parsed JSON data to initialize devices.
        - Increments the total device count (`n`) for each model.
        - Assigns parameter values to their corresponding model attributes in a vectorized manner.

        Args:
            data (dict): A dictionary where keys are model names, and values are lists of device data.
        """

        for model_name, device_list in data.items():
            # Retrieve the corresponding model instance
            model = self.models[model_name]  
            # Save system devices 
            self.devices[model_name] = model

            for device in device_list:
                # Increment the count of devices for this model
                model.n += 1  
                
                for param_name, value in device.items():
                    if hasattr(model, param_name):
                        param = getattr(model, param_name)

                        # Store parameter values in the appropriate structure: either IdxDynParam or NumDynParam
                        if isinstance(param, IdxDynParam):
                            param.id.append(value)
                        elif isinstance(param, NumDynParam):
                            param.value.append(value)


    def finalize_pycode(self):
        """
        Generates an __init__.py file to later dynamically import compiled models.

        This method:
        - Retrieves the path of the generated Python code directory.
        - Writes import statements for each model into __init__.py.
        - Compiles all Python files within the directory to optimize execution (.pyc).
        """

        pycode_path = get_pycode_path()
        init_path = os.path.join(pycode_path, '__init__.py')

        # Write import statements for dynamically generated model files
        with open(init_path, 'w') as f:
            for model_name in self.models.keys():
                #Import each model dynamically
                f.write(f"from . import {model_name}\n")

        # Compile all generated Python code to bytecode (.pyc)
        compileall.compile_dir(pycode_path)

######################### TO CLEAN #######################################
    def store_params(self):
        for device in self.devices.values():
            for param_name, param in device.dict.items():
                if isinstance(param, NumDynParam):
                    self.dae.params_dict[device.name][param.symbol] = param.value

    def set_addresses(self):
        self.global_id = 0
        algeb_ref_map = {}  # Cache: store algeb_idx references for quick lookup
        states_ref_map = {}  # Cache: store states_idx references for quick lookup

        # First loop: Process StatesVar
        for model_instance in self.devices.values():
            for var_list in model_instance.__dict__.values():
                if isinstance(var_list, StatVar):
                    indices = list(range(self.global_id, self.global_id + model_instance.n))
                    model_instance.states_idx[var_list.name] = indices
                    self.global_id += model_instance.n  # Move global index forward

                    # Cache reference for faster lookup
                    states_ref_map[(model_instance.__class__.__name__, var_list.name)] = indices

                    # Store DAE addresses
                    self.dae.xy_addr.extend(indices)
                    if model_instance.name != 'Bus':
                        self.dae.x_addr.extend(indices)

                    self.dae.nx += model_instance.n

        # Second loop: Process ExternStates
        for model_instance in self.devices.values():
            for var_list in model_instance.__dict__.values():
                if isinstance(var_list, ExternState):
                    key = (var_list.indexer.symbol, var_list.src)

                    if key not in states_ref_map:
                        raise KeyError(f"Variable '{var_list.src}' not found in {var_list.indexer.symbol}.algeb_idx")

                    parent_idx = states_ref_map[key]

                    # Store in extstates_idx using src as the key (grouping multiple references)
                    if var_list.name not in model_instance.extstates_idx:
                        model_instance.extstates_idx[var_list.name] = []  # Initialize as a list of lists

                    model_instance.extstates_idx[var_list.name] = [parent_idx[i] for i in var_list.indexer.id]

                    # Store DAE addresses
                    self.dae.xy_addr.extend(model_instance.extalgeb_idx[var_list.name])
                    self.dae.x_addr.extend(model_instance.extalgeb_idx[var_list.name])

        # First loop: Process AlgebVar
        for model_instance in self.devices.values():
            for var_list in model_instance.__dict__.values():
                if isinstance(var_list, AlgebVar):
                    indices = list(range(self.global_id, self.global_id + model_instance.n))
                    model_instance.algeb_idx[var_list.name] = indices
                    self.global_id += model_instance.n  # Move global index forward
                    
                    # Cache reference for faster lookup
                    algeb_ref_map[(model_instance.__class__.__name__, var_list.name)] = indices

                    # Store DAE addresses
                    self.dae.xy_addr.extend(indices)
                    if model_instance.name != 'Bus':
                        self.dae.y_addr.extend(indices)

                    self.dae.ny += model_instance.n  
         
        # Second loop: Process ExternAlgeb
        for model_instance in self.devices.values():
            for var_list in model_instance.__dict__.values():
                if isinstance(var_list, ExternAlgeb):
                    key = (var_list.indexer.symbol, var_list.src)

                    if key not in algeb_ref_map:
                        raise KeyError(f"Variable '{var_list.src}' not found in {var_list.indexer.symbol}.algeb_idx")

                    parent_idx = algeb_ref_map[key]  

                    # Store in extalgeb_idx using src as the key (grouping multiple references)
                    if var_list.name not in model_instance.extalgeb_idx:
                        model_instance.extalgeb_idx[var_list.name] = []  # Initialize as a list of lists

                    model_instance.extalgeb_idx[var_list.name] = [parent_idx[i] for i in var_list.indexer.id]

                    # Store DAE addresses
                    self.dae.xy_addr.extend(model_instance.extalgeb_idx[var_list.name])
                    self.dae.y_addr.extend(model_instance.extalgeb_idx[var_list.name])

    def build_input_dict(self):
        
        index1 = 0
        for model_instance in self.devices.values():
            if model_instance.name != 'Bus':
                nr_components = model_instance.n
                for variable in model_instance.variables_list:
                    values = (self.values_array[index1:index1+nr_components])
                    self.dae.residuals_dict[model_instance.name][variable] = values
                    index1 += nr_components


    def get_input_values(self, device):

        #get parameters and residuals from "dae"
        self.build_input_dict()

        residuals = self.dae.residuals_dict[device.name]
        parameters= self.dae.params_dict[device.name]
        parameters.update(residuals)

        pycode_code = device.import_pycode()
        f_jac_arguments = pycode_code.f_jac_args
        g_jac_arguments = pycode_code.g_jac_args

        f_arguments = pycode_code.f_args
        g_arguments = pycode_code.g_args

        # create input values lists
        f_input_values = [parameters[argument] for argument in f_arguments]
        g_input_values = [parameters[argument] for argument in g_arguments]
        f_input_values = list(zip(*f_input_values))
        g_input_values = list(zip(*g_input_values))

        f_jac_input_values = [parameters[argument] for argument in f_jac_arguments]
        g_jac_input_values = [parameters[argument] for argument in g_jac_arguments]
        f_jac_input_values = list(zip(*f_jac_input_values))
        g_jac_input_values = list(zip(*g_jac_input_values))


        return f_input_values, g_input_values, f_jac_input_values, g_jac_input_values
    

    def update_jacobian(self):
        all_triplets = {}
        f_values_list = []
        g_values_list = []
        #all_f_g_values = []
        for device in self.devices.values():

            f_input_values, g_input_values, f_jac_input_values, g_jac_input_values = self.get_input_values(device)

            # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template
            if device.name != 'Bus':
                # get f and g update
                f_values, g_values = device.calc_f_g_functions(f_input_values, g_input_values)
                f_values_list.append(f_values)
                g_values_list.append(g_values)

                # get local jacobians info and values
                f_jacobians, g_jacobians, jacobian_info = device.calc_local_jacs(f_jac_input_values, g_jac_input_values)

                # get variable addresses
                g_var_addresses = device.extalgeb_idx
                g_var_addresses.update(device.algeb_idx)

                f_var_addresses = device.extstates_idx
                f_var_addresses.update(device.states_idx)

                f_var_addresses.update(g_var_addresses)
                var_addresses = f_var_addresses

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_positions(device, f_jacobians, jac_type, positions, var_addresses)
                all_triplets[jac_type] = triplets
                for row, col, val in triplets:
                    self.dae.add_to_jacobian(self.dae._dfx_dict, self.dae.sparsity_fx, row, col, val)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_positions(device, f_jacobians, jac_type, positions, var_addresses)
                all_triplets[jac_type] = triplets
                for row, col, val in triplets:
                    self.dae.add_to_jacobian(self.dae._dfy_dict, self.dae.sparsity_fy, row, col, val)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_positions(device, g_jacobians, jac_type, positions, var_addresses)
                all_triplets[jac_type] = triplets
                for row, col, val in triplets:
                    self.dae.add_to_jacobian(self.dae._dgx_dict, self.dae.sparsity_gx, row, col, val)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_positions(device, g_jacobians, jac_type, positions, var_addresses)
                all_triplets[jac_type] = triplets
                for row, col, val in triplets:
                    self.dae.add_to_jacobian(self.dae._dgy_dict, self.dae.sparsity_gy, row, col, val)
            
        #all_f_g_values = [f_values_list, g_values_list]

        # NOTE: necessary?
        f_value_list_flat = [val for model in f_values_list for val in model]
        g_value_list_flat = [val for model in g_values_list for val in model]

        f_array = np.array(f_value_list_flat)

        # Residual power balance
        g_array = self.sum_duplicate_values(g_value_list_flat)

        self.dae.f = f_array
        self.dae.g = g_array

        return all_triplets


    def assign_positions(self, model, local_jacobian, jac_type, positions, var_addresses):
        triplets = []
        for i in range(model.n):

            for j, (func_index, var_index) in enumerate(positions):

                val = local_jacobian[i][j]
                address_func = var_addresses[model.vars_index[func_index]][i]
                address_var = var_addresses[model.vars_index[var_index]][i]
                triplets.append((address_func, address_var, val))

        return triplets
    
    def sum_duplicate_values(self, g_value_list_flat):
        """
        Sum values with the same address, keeping only the first occurrence in order.
        
        Parameters:
        - addresses: iterable of int
        - values: iterable of float
        
        Returns:
        - summed_values: list of float, same order as first appearance of each unique address
        - unique_addresses: list of int, order preserved
        """
        addresses = np.array(self.dae.y_addr)
        values = np.array(g_value_list_flat).flatten()

        seen = {}
        result = []
        
        for addr, val in zip(addresses, values):
            if addr in seen:
                result[seen[addr]] += val
            else:
                seen[addr] = len(result)
                result.append(val)
        
        return np.array(result)

