# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from GridCalEngine.Devices.Dynamic.dynamic_model import DynamicModel
from GridCalEngine.Simulations.Dynamic.model.symbolic_process import SymProcess
from GridCalEngine.IO.file_system import (get_create_dynamics_model_folder, load_file_as_module,
                                          load_function_from_module)


class RmsModelStore:

    def __init__(self, dynamic_model: DynamicModel, grid_id: str):
        """

        :param dynamic_model:
        :param grid_id:
        """
        self.name = dynamic_model.name

        # Device index for ordering devices in the system (used to relate variables with their addresses)
        self.index = int

        # Set address function
        self.n = 0  # index for the number of components corresponding to this model

        # Symbolic processing engine utils
        self.stats = list()
        self.algebs = list()

        # dictionary containing index of the variable as key and symbol of the variable as value
        self.vars_index = np.empty(dynamic_model.get_var_num(), dtype=object)

        # Lists of internal and external variables
        self.internal_vars = list()
        self.external_vars = list()

        # list containing all the symbols of the variables in the model (used in f, g, and jacobian calculation)
        self.variables_list = list()  # list of all the variables (including external)
        self.state_eqs = list()
        self.state_vars = list()
        self.algeb_eqs = list()
        self.algeb_vars = list()
        self.eqs_list = list()  # list of all the variables with an equation
        self.vars_list = list()  # list of all the variables

        # Lists to store function arguments
        self.f_args = list()
        self.g_args = list()

        self.f_jac_arguments = list()
        self.g_jac_arguments = list()
        self.jacobian_info = {}

        # Lists to store input values
        self.f_input_values = np.zeros((self.n, len(self.f_args)), dtype=object)
        self.g_input_values = np.zeros((self.n, len(self.g_args)), dtype=object)
        self.f_jac_input_values = np.zeros((self.n, len(self.f_jac_arguments)), dtype=object)
        self.g_jac_input_values = np.zeros((self.n, len(self.g_jac_arguments)), dtype=object)

        # Lists to store inputs order when updating f, g, and jacobian functions

        self.g_inputs_order = list()
        self.f_inputs_order = list()
        self.f_jac_inputs_order = list()
        self.g_jac_inputs_order = list()
        self.f_output_order = list()
        self.g_output_order = list()
        self.dfx_jac_output_order = list()
        self.dfy_jac_output_order = list()
        self.dgx_jac_output_order = list()
        self.dgy_jac_output_order = list()

        self.time_consuming = []

        # index for states and algebraic variables (used to compute dae.nx and dae.ny)
        self.nx = 0  # index for the number of state variables (not external)
        self.ny = 0  # index for the number of algebraic variables (not external)

        """
        Fill in the structures
        """

        index = 0

        for key, elem in dynamic_model.stat_var.items():
            self.variables_list.append(elem.symbol)
            self.vars_index[index] = elem.symbol
            index += 1

            self.nx += 1
            if elem.eq is not None:
                self.state_eqs.append(elem)
                self.eqs_list.append(elem.symbol)
            self.state_vars.append(elem)
            self.vars_list.append(elem.symbol)
            self.internal_vars.append(elem.symbol)

        for key, elem in dynamic_model.algeb_var.items():
            self.variables_list.append(elem.symbol)
            self.vars_index[index] = elem.symbol
            index += 1

            self.ny += 1
            if elem.eq is not None:
                self.algeb_eqs.append(elem)
                self.eqs_list.append(elem.symbol)
            self.algeb_vars.append(elem)
            self.vars_list.append(elem.symbol)
            self.internal_vars.append(elem.symbol)

        for key, elem in dynamic_model.ext_state_var.items():
            self.variables_list.append(elem.symbol)
            self.vars_index[index] = elem.symbol
            index += 1

            if elem.eq is not None:
                self.state_eqs.append(elem)
                self.eqs_list.append(elem.symbol)
            self.state_vars.append(elem)
            self.vars_list.append(elem.symbol)
            self.external_vars.append(elem)

        for key, elem in dynamic_model.ext_algeb_var.items():
            self.variables_list.append(elem.symbol)
            self.vars_index[index] = elem.symbol
            index += 1

            if elem.eq is not None:
                self.algeb_eqs.append(elem)
                self.eqs_list.append(elem.symbol)
            self.algeb_vars.append(elem)
            self.vars_list.append(elem.symbol)
            self.external_vars.append(elem)

        folder_to_save = get_create_dynamics_model_folder(grid_id=grid_id)
        sym_process = SymProcess(dynamic_model=self, folder_to_save=folder_to_save)
        file_path = sym_process.generate() # does all the symbolic operations needed

        self._variables_names_for_ordering = sym_process.variables_names_for_ordering
        self._jacobian_info = sym_process.jacobian_store_info
        self._jacobian_equations = sym_process.jacobian_store_equations
        self._variables_names_for_ordering_f = self._variables_names_for_ordering['f']
        self._variables_names_for_ordering_g = self._variables_names_for_ordering['g']

        self._module = load_file_as_module(file_path=file_path)
        self._f_update_ptr = load_function_from_module(module=self._module, function_name="f_update")
        self._g_update_ptr = load_function_from_module(module=self._module, function_name="g_update")
        self._f_ia_ptr = load_function_from_module(module=self._module, function_name="f_ia")
        self._g_ia_ptr = load_function_from_module(module=self._module, function_name="g_ia")

    def calc_f_g_functions(self):
        """
        calculates f and g functions and gets variables order.
        :return: f and g functions values, lists with the order of the variables for f and g
        """

        f_values_device = np.zeros((self.n, len(self.state_eqs)))
        g_values_device = np.zeros((self.n, len(self.algeb_eqs)))

        for i in range(self.n):
            # get f values
            if self.f_args:
                f_values = self._f_update_ptr(*self.f_input_values[i])
                for j in range(len(self.state_eqs)):
                    f_values_device[i][j] = f_values[j]

            # get g values
            if self.g_args:
                g_values = self._g_update_ptr(*self.g_input_values[i])
                for j in range(len(self.algeb_eqs)):
                    g_values_device[i][j] = g_values[j]

        return (f_values_device, g_values_device,
                self._variables_names_for_ordering_f, self._variables_names_for_ordering_g)

    def calc_local_jacs(self):
        """

        :return:
        """
        f_jacobians = np.zeros((self.n, len(self.state_eqs), len(self.variables_list)))
        g_jacobians = np.zeros((self.n, len(self.algeb_eqs), len(self.variables_list)))

        for i in range(self.n):

            if self.f_jac_arguments:
                local_jac_f = self._f_ia_ptr(*self.f_jac_input_values[i])
                for j, funct in enumerate(self.state_eqs):
                    for k, var in enumerate(self.variables_list):
                        f_jacobians[i, j, k] = local_jac_f[j * len(self.variables_list) + k]

            if self.g_jac_arguments:
                local_jac_g = self._g_ia_ptr(*self.g_jac_input_values[i])
                for j, funct in enumerate(self.algeb_eqs):
                    for k, var in enumerate(self.variables_list):
                        g_jacobians[i, j, k] = local_jac_g[j * len(self.variables_list) + k]

        jacobian = np.concatenate((f_jacobians, g_jacobians), axis=1)

        return jacobian, self._jacobian_info, self._jacobian_equations
