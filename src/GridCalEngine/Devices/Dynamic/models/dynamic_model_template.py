# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import importlib
import numpy as np
from GridCalEngine.Devices.Parents.editable_device import EditableDevice, DeviceType
from typing import Union
from GridCalEngine.Devices.Dynamic.utils.paths import get_generated_module_path
from GridCalEngine.Devices.Dynamic.symprocess import SymProcess
from GridCalEngine.Utils.dyn_var import *


class DynamicModelTemplate(EditableDevice):
    """
    Represents a dynamic model template for a device, handling symbolic processing,
    storage of variables, and setting addresses.

    Inherits from EditableDevice, allowing dynamic model creation and symbolic processing.
    """

    def __init__(self, name: str, code: str, idtag: Union[str, None],
                 device_type: DeviceType):
        """
        DynamicModelTemplate class constructor. Initializes a dynamic model template with symbolic processing and storage.

        :param name: Name of the dynamic model.
        :param code: Unique code identifier.
        :param idtag: Optional tag for identifying the model instance.
        :param device_type: The type of the device (e.g., generator, load, etc.).
        """
        EditableDevice.__init__(self,
                                name=name,
                                code=code,
                                idtag=idtag,
                                device_type=device_type)

        # Device index for ordering devices in the system (used to relate variables with their addresses)
        self.index = int

        # Set address function
        self.n = 0  # index for the number of components corresponding to this model

        # Symbolic processing engine utils
        self.sym = SymProcess(self)
        self.stats = []
        self.algebs = []

        # dictionary containing index of the variable as key and symbol of the variable as value
        self.vars_index = {}

        # list containing all the symbols of the variables in the model (used in f, g, and jacobian calculation)
        self.variables_list = []  # list of all the variables (including external)
        self.state_vars_list = []  # list of all the state variables (including external)
        self.algeb_vars_list = []  # list of all the algebraic variables (including external)

        # Lists to store function arguments
        self.f_args = list()
        self.g_arguments = list()

        self.f_jac_arguments = list()
        self.g_jac_arguments = list()

        # Lists to store input values
        self.g_input_values = list()
        self.f_input_values = list()
        self.f_jac_input_values = list()
        self.g_jac_input_values = list()

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

    def process_symbolic(self):
        """
        Generates symbolic equations and Jacobians for the dynamic model.
        :return:
        """
        self.sym.generate()

    def store_data(self):
        """
        Stores different types of variables and parameters.
        This method categorizes each instance variable and adds it to the corresponding
        storage structure.
        Also, it saves a list with all the variables of a model and creates a dictionary with an index as key and the variable name as value
        :return:
        """
        index = 0
        for key, elem in self.__dict__.items():
            # assign an index to every variable in the model populating vars_index dictionary
            if isinstance(elem, DynVar):
                self.variables_list.append(elem.symbol)
                self.vars_index[index] = elem.symbol
                index += 1

            if isinstance(elem, StatVar):
                self.nx += 1
                self.state_vars_list.append(elem.symbol)
                self.stats.append(elem)

            if isinstance(elem, AlgebVar):
                self.ny += 1
                self.algeb_vars_list.append(elem.symbol)
                self.algebs.append(elem)

            if isinstance(elem, ExternState):
                self.state_vars_list.append(elem.symbol)
                self.stats.append(elem)

            if isinstance(elem, ExternAlgeb):
                self.algeb_vars_list.append(elem.symbol)
                self.algebs.append(elem)

    ####################### TO CLEAN ################################
    def import_generated_code(self):
        """
        Imports generated code module
        :return: generated code module
        """
        generated_module_path = get_generated_module_path()
        generated_module = importlib.import_module(generated_module_path.replace("/", "."))
        generated_code = getattr(generated_module, self.name)

        return generated_code

    def calc_f_g_functions(self):
        """
        calculates f and g functions and gets variables order.
        :return: f and g functions values, lists with the order of the variables for f and g
        """
        generated_code = self.import_generated_code()
        f_values_device = np.zeros((self.n, len(self.state_vars_list)))
        g_values_device = np.zeros((self.n, len(self.algeb_vars_list)))
        for i in range(self.n):
            # get f values
            if self.f_input_values[i]:
                f_values = generated_code.f_update(*self.f_input_values[i])
                for j in range(len(self.state_vars_list)):
                    f_values_device[i][j] = f_values[j]

            # get g values
            if self.g_input_values[i]:
                g_values = generated_code.g_update(*self.g_input_values[i])
                for j in range(len(self.algeb_vars_list)):
                    g_values_device[i][j] = g_values[j]

        variables_names_for_ordering_f = generated_code.variables_names_for_ordering['f']
        variables_names_for_ordering_g = generated_code.variables_names_for_ordering['g']

        return f_values_device, g_values_device, variables_names_for_ordering_f, variables_names_for_ordering_g

    def calc_local_jacs(self):
        """
        calculates f and g Jacobians and gets Jacobians info.
        :return: f Jacobian, g Jacobian, Jacobians info
        """
        generated_code = self.import_generated_code()
        jacobian_info = generated_code.jacobian_info
        jacobian_equations = generated_code.jacobian_equations

        f_jacobians = np.zeros((self.n, len(self.state_vars_list), len(self.variables_list)))
        g_jacobians = np.zeros((self.n, len(self.variables_list), len(self.variables_list)))
        for i in range(self.n):
            if self.f_jac_input_values[i]:
                local_jac_f = generated_code.f_ia(*self.f_jac_input_values[i])
                for j, funct in enumerate(self.state_vars_list):
                    for k, var in enumerate(self.variables_list):
                        f_jacobians[i][j][k] = local_jac_f[j * len(self.variables_list) + k]
            if self.g_jac_input_values[i]:
                local_jac_g = generated_code.g_ia(*self.g_jac_input_values[i])
                for j, funct in enumerate(self.algeb_vars_list):
                    for k, var in enumerate(self.variables_list):
                        g_jacobians[i][j + len(self.state_vars_list)][k] = local_jac_g[j * len(self.variables_list) + k]
        return f_jacobians, g_jacobians, jacobian_info, jacobian_equations
