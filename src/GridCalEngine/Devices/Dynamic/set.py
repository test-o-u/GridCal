# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import importlib
import pdb
import time
import logging
import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Devices.Dynamic.utils.paths import get_generated_module_path


class SET:
    """
    System set-up class

    Responsible for:
        - Importing and initializing dynamic models
        - Processing models symbolically and generating fast numerical code
        - Creating and configuring system devices
        - Assigning addresses for variables used in simulations
    """

    def __init__(self, system, models_list, data):
        """
        SET class constructor
        initializes the SET class and prepares the system.
        :param system: the system instance containing models and devices
        :param models_list: List of tuples, each containing a category and a list of model names
        :param data: Parsed JSON structure where keys are model names and values are lists of device data

        """

        self.system = system
        self.dae = self.system.dae

        self.models = self.system.models
        self.devices = self.system.devices

        self.models_list = models_list
        self.data = data

        self.import_models()
        self.system_prepare()

    def import_models(self):
        """
        Imports dynamic models and initializes their symbolic-numeric representations.
        This method:
        - Dynamically imports model classes from GridCalEngine
        - Creates empty instances of each model and stores them in `self.models`
        :return:
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
        1. Processing models symbolically and generating optimized numerical functions
        2. Creating instances of devices and storing them in vectorized form
        3. Assigning global indices to algebraic variables and external references
        :return:
        """

        start_time = time.perf_counter()

        # Step 1: Process models symbolically and generate numerical functions
        symb_st = time.perf_counter()
        for model in self.models.values():
            model.store_data()
            model.process_symbolic()
        self.finalize_generated_code()  # Finalize generated code
        logging.info(f"Generated code module created")
        symb_end = time.perf_counter()
        symb_time = symb_end - symb_st  # Store symbolic processing time

        # Step 2: Create vectorized model instances for device storage
        dev_st = time.perf_counter()
        self.create_devices(self.data)
        dev_end = time.perf_counter()
        dev_time = dev_end - dev_st  # Store device creation time

        # Step 3: Store parameters and assign global indices to variables and external references
        add_st = time.perf_counter()
        self.set_addresses()
        # demo
        logging.info(f"Addresses created for {self.dae.nx+self.dae.ny} internal variables")
        logging.info(f"Printing list of variables and addresses" + '\n\n' + f"Variables: {self.dae.variables_list} " + '\n\n' + f"Addresses: {self.dae.addresses_list} ")

        add_end = time.perf_counter()
        add_time = add_end - add_st  # Store addressing time

        # Log time performance
        if config.PERFORMANCE:
            logging.info("=============== TIME CHECK ================")
            logging.info(f"Process symbolic time = {symb_time:.6f} [s]")
            logging.info(f"Create device time = {dev_time:.6f} [s]")
            logging.info(f"Set address time = {add_time:.6f} [s]")
            total_time = time.perf_counter() - start_time
            logging.info(f"Total execution time: {total_time:.6f} [s]")
            logging.info("===========================================")

    def finalize_generated_code(self):
        """
        Generates __init__.py to dynamically import compiled model modules.

        This method:
        - Writes import statements for each model into __init__.py
        :return:
        """
        generated_module_path = get_generated_module_path()
        init_path = os.path.join(generated_module_path, '__init__.py')

        # Write import statements for dynamically generated model files
        with open(init_path, 'w') as f:
            for model_name in self.models.keys():
                # Import each model dynamically
                f.write(f"from . import {model_name}\n")

    def create_devices(self, data):
        """
        Populates vectorized model instances with device data from a parsed JSON file.
        :param data: A dictionary with model names as keys and lists of device data as values
        :return:
        """

        device_index = 0
        for model_name, device_list in data.items():

            # Retrieve the corresponding model instance
            model = self.models[model_name]
            # Save system devices
            self.devices[model_name] = model
            # set device index
            model.index = device_index
            device_index += 1

            # Save arguments for the updating functions
            generated_code = model.import_generated_code()

            model.f_args = generated_code.f_args
            model.g_args = generated_code.g_args

            model.f_jac_args = generated_code.f_jac_args
            model.g_jac_args = generated_code.g_jac_args

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

            # calculate nx and ny and save it in dae
            self.dae.nx += model.n * model.nx
            self.dae.ny += model.n * model.ny

            # Initialize the lists that will store the input values of the equations corresponding to this device
            model.g_input_values = [[] for i in range(model.n)]
            model.f_input_values = [[] for i in range(model.n)]
            model.f_jac_input_values = [[] for i in range(model.n)]
            model.g_jac_input_values = [[] for i in range(model.n)]

    def set_addresses(self):
        """
        Assign global DAE indices to variables, store parameter values and store a reference for results analysis.
        This method:
            - Assigning local and global indices to state and algebraic variables
            - Mapping external variable references
            - Populating the `dae.addresses_dict` and `dae.params_dict`
            - Storing a reference list to analyse results
        :return:
        """

        self.global_states_id = 0
        self.global_algebs_id = self.dae.nx
        algeb_ref_map = {}  # Cache: store algeb_idx references for quick lookup
        states_ref_map = {}  # Cache: store states_idx references for quick lookup

        # Loop through devices
        for model_instance in self.devices.values():
            # initialize variables list and addresses list for this device
            device_variables_list = list()
            device_addresses_list = list()

            # Store parameters and assign addresses
            for var_list in model_instance.__dict__.values():

                # state varibles first
                if isinstance(var_list, StatVar):
                    indices = list(range(self.global_states_id, self.global_states_id + model_instance.n))

                    # store internal variables in dae
                    for i in range(model_instance.n):
                        self.dae.internal_variables_list.append(
                            (indices[i], var_list.symbol + '_' + model_instance.name + '_' + str(i)))

                    # store variables names and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append(indices)

                    self.global_states_id += model_instance.n  # Move global index forward

                    # Cache reference for faster lookup
                    states_ref_map[(model_instance.__class__.__name__, var_list.name)] = indices

                    # Construct DAE lhs matrix
                    if var_list.t_const != 1.0:
                        self.dae.Tf += var_list.t_const
                    else:
                        self.dae.Tf += [1.0] * model_instance.n

                # external state variables
                if isinstance(var_list, ExternState):
                    key = (var_list.indexer.symbol, var_list.src)

                    if key not in states_ref_map:
                        raise KeyError(f"Variable '{var_list.src}' not found in {var_list.indexer.symbol}.algeb_idx")

                    parent_idx = states_ref_map[key]

                    # store variable name and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append([parent_idx[i] for i in var_list.indexer.id])

                # algebraic variables second
                if isinstance(var_list, AlgebVar):
                    indices = list(range(self.global_algebs_id, self.global_algebs_id + model_instance.n))

                    # store internal variables in dae
                    for i in range(model_instance.n):
                        self.dae.internal_variables_list.append(
                            (indices[i], var_list.symbol + '_' + model_instance.name + '_' + str(i)))

                    self.global_algebs_id += model_instance.n  # Move global index forward

                    # store variable name and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append(indices)

                    # Cache reference for faster lookup
                    algeb_ref_map[(model_instance.__class__.__name__, var_list.name)] = indices

                # external algebraic variables
                if isinstance(var_list, ExternAlgeb):
                    key = (var_list.indexer.symbol, var_list.src)

                    if key not in algeb_ref_map:
                        raise KeyError(f"Variable '{var_list.src}' not found in {var_list.indexer.symbol}.algeb_idx")

                    parent_idx = algeb_ref_map[key]

                    # store variable name and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append([parent_idx[i] for i in var_list.indexer.id])

            # add variables names local list and addresses local list to dae general lists
            self.dae.variables_list.append(device_variables_list)
            self.dae.addresses_list.append(device_addresses_list)