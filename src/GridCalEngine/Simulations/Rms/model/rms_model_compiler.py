# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import List
import os
import importlib
import pdb
import time
import logging
from GridCalEngine.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Simulations.Rms.model.rms_model_store import RmsModelStore
from GridCalEngine.Devices.Dynamic.dyn_param import NumDynParam, IdxDynParam, ExtDynParam
from GridCalEngine.Devices.Dynamic.dyn_var import StatVar, AlgebVar, InputState, InputAlgeb


class RmsModelsCompiler:
    """
    System set-up class

    Responsible for:
        - Importing and initializing dynamic models
        - Processing models symbolically and generating fast numerical code
        - Creating and configuring system devices
        - Assigning addresses for variables used in simulations
    """

    def __init__(self, grid: MultiCircuit):
        """
        SET class constructor
        initializes the SET class and prepares the system.
        :param grid: MultiCircuit

        """

        self.grid = grid
        self.models_list: List[RmsModelStore] = list()

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
        # symb_st = time.perf_counter()
        # for model in self.models.values():
        #     model.store_data()
        #     model.process_symbolic()
        # self.finalize_generated_code()  # Finalize generated code
        # logging.info(f"Generated code module created")
        # symb_end = time.perf_counter()
        # symb_time = symb_end - symb_st  # Store symbolic processing time

        # Step 2: Create vectorized model instances for device storage
        # dev_st = time.perf_counter()
        # # self.create_devices(self.data)
        # self.create_devices()
        # self.devices.move_to_end('Bus', last=False)
        # dev_end = time.perf_counter()
        # dev_time = dev_end - dev_st  # Store device creation time

        # Step 3: Store parameters and assign global indices to variables and external references
        add_st = time.perf_counter()
        self.set_addresses()
        # demo
        logging.info(f"Addresses created for {self.dae.nx + self.dae.ny} internal variables")
        logging.info(
            f"Printing list of variables and addresses" + '\n\n' + f"Variables: {self.dae.variables_list} " + '\n\n' + f"Addresses: {self.dae.addresses_list} ")

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

    def create_devices_gridcal_int(self, gridcal_data):
        """
        Populates vectorized model instances with device data from a parsed JSON file.
        :param data: A dictionary with model names as keys and lists of device data as values
        :return:
        """
        pflow_dev = ['Bus', 'ACLine', 'ExpLoad', 'Slack']
        sim_dev = ['Bus', 'ACLine', 'ExpLoad', 'GENCLS']

        device_index = 1
        for family in gridcal_data:
            for device in family:
                if device["name"] in sim_dev:

                    # Retrieve the corresponding model instance
                    model = self.models[device["name"]]
                    # Save system devices
                    self.devices[device["name"]] = model
                    # set device index
                    if device["name"] == "Bus":
                        model.index = 0
                    else:
                        model.index = device_index
                        device_index += 1

                    # Save arguments for the updating functions
                    generated_code = model.import_generated_code()

                    model.f_args = generated_code.f_args
                    model.g_args = generated_code.g_args

                    model.f_jac_args = generated_code.f_jac_args
                    model.g_jac_args = generated_code.g_jac_args
                    model.jacobian_info = generated_code.jacobian_info
                    self.dae.ndfx += len(model.jacobian_info['dfx'])
                    self.dae.ndfy += len(model.jacobian_info['dfy'])
                    self.dae.ndgx += len(model.jacobian_info['dgx'])
                    self.dae.ndgy += len(model.jacobian_info['dgy'])

                    for attribute_type_list in device.items():

                        if attribute_type_list[0] in ("idx_dyn_param", "num_dyn_param"):
                            for attribute in attribute_type_list[1]:

                                if hasattr(model, attribute["name"]):
                                    param = getattr(model, attribute["name"])

                                    # Store parameter values in the appropriate structure: either IdxDynParam or NumDynParam
                                    if isinstance(param, IdxDynParam):
                                        param.id = attribute["id"]
                                        param.connection_point = attribute["connection_point"]
                                    if isinstance(param, NumDynParam):
                                        param.value = attribute["value"]
                    model.n = len(device["comp_name"])
                    # calculate nx and ny and save it in dae
                    self.dae.nx += model.n * model.nx
                    self.dae.ny += model.n * model.ny

    def create_devices(self):
        """
        Populates vectorized model instances with device data from a parsed JSON file.
        :param data: A dictionary with model names as keys and lists of device data as values
        :return:
        """
        pflow_dev = ['Bus', 'ACLine', 'ExpLoad', 'Slack']
        sim_dev = ['Bus', 'ACLine', 'ExpLoad', 'GENCLS']

        device_index = 0
        for model in self.system.models():

            if model.name in sim_dev:
                # Save system devices
                self.devices[model.name] = model
                # set device index
                model.index = device_index
                device_index += 1

                # Save arguments for the updating functions
                generated_code = model.import_generated_code()

                model.f_args = generated_code.f_args
                model.g_args = generated_code.g_args

                model.f_jac_args = generated_code.f_jac_args
                model.g_jac_args = generated_code.g_jac_args
                model.jacobian_info = generated_code.jacobian_info
                self.dae.ndfx += len(model.jacobian_info['dfx'])
                self.dae.ndfy += len(model.jacobian_info['dfy'])
                self.dae.ndgx += len(model.jacobian_info['dgx'])
                self.dae.ndgy += len(model.jacobian_info['dgy'])

                model.n = 1
                # calculate nx and ny and save it in dae
                self.dae.nx += model.n * model.nx
                self.dae.ny += model.n * model.ny

    def set_addresses(self, models_list):
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
        for model_store in models_list:
            # initialize variables list and addresses list for this device
            device_variables_list = list()
            device_addresses_list = list()

            # Assign addresses
            for var_list in model_store.variables:

                # state varibles first
                if isinstance(var_list, StatVar):
                    address = self.global_states_id
                    var_list.address = address

                    # store variables names and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append(address)


                    self.global_states_id += 1  # Move global index forward

                if isinstance(var_list, InputState):

                    # store variable name and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append(var_list.address)

                # algebraic variables second
                if isinstance(var_list, AlgebVar):
                    address = self.global_algebs_id
                    var_list.address = address

                    # store variable name and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append(address)

                    self.global_algebs_id += 1  # Move global index forward


                # external algebraic variables
                if isinstance(var_list, InputAlgeb):

                    # store variable name and addresses locally
                    device_variables_list.append(var_list.symbol)
                    device_addresses_list.append(var_list.address)

            # add variables names local list and addresses local list to dae general lists
            self.dae.variables_list.append(device_variables_list)
            self.dae.addresses_list.append(device_addresses_list)

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

                # check device connection to the grid
                if isinstance(var_list, IdxDynParam):
                    connection_element = var_list.symbol
                    connection_point = var_list.connection_point
                    connection_id = connection_point + '_' + connection_element
                    connecting_vars = [var.src for var in model_instance.external_vars if
                                       var.indexer.symbol == connection_element and var.indexer.connection_point == connection_point]
                    if all(var in self.devices[connection_element].internal_vars for var in connecting_vars):
                        # if connecting_vars == self.devices[connection_element].internal_vars:
                        self.system.connections.append(connection_id)
                    else:
                        model_instance.u = [0] * model_instance.n
                        self.system.connections.append(connection_id)
                    pdb.set_trace()

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
                    states_ref_map[(model_instance.name, var_list.name)] = indices

                    # Construct DAE lhs matrix
                    if var_list.t_const != 1.0:
                        self.dae.Tf += var_list.t_const
                    else:
                        self.dae.Tf += [1.0] * model_instance.n

                # external state variables
                if isinstance(var_list, InputState):
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
                    algeb_ref_map[(model_instance.name, var_list.name)] = indices

                # external algebraic variables
                if isinstance(var_list, InputAlgeb):
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
