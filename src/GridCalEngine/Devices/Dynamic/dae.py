# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb
import sympy as sp
from sympy.matrices import zeros, ones
from sympy import preview
import logging


import numpy as np
import GridCalEngine.Devices.Dynamic.io.config as config
from scipy.sparse import coo_matrix, diags
from collections import defaultdict


class DAE:
    """
    DAE (Differential-Algebraic Equation) class to store and manage.

    Responsibilities:
        - Store state and algebraic variables (x, y)
        - Store Jacobian matrices
        - Store residual equations
        - Store sparsity patterns
    """

    def __init__(self, system):
        """
        DAE class constructor
        Initialize DAE object with required containers and defaults.
        :param system: The simulation system object to which this DAE is tied
        """

        self.system = system

        # List to store variables internal variables of the system (used to analyze simulation data)
        self.internal_variables_list = list()

        # List to store all variables (used when updating f, g, and jacobian values)
        self.variables_list = list()

        # List to store all addresses (used when updating f, g, and jacobian values)
        self.addresses_list = list()

        self.nx = 0
        self.ny = 0

        self.x = config.DAE_X0
        self.y = config.DAE_Y0
        self.xy = None

        # Lists to accumulate Jacobian positions, values and equations

        self._dfx_jac_positions = list()
        self._dfy_jac_positions = list()
        self._dgx_jac_positions = list()
        self._dgy_jac_positions = list()

        self._dfx_jac_values = list()
        self._dfy_jac_values = list()
        self._dgx_jac_values = list()
        self._dgy_jac_values = list()

        self._dfx_jac_equ = list()
        self._dfy_jac_equ = list()
        self._dgx_jac_equ = list()
        self._dgy_jac_equ = list()

        # Sparse Jacobian values
        self.dfx = None
        self.dfy = None
        self.dgx = None
        self.dgy = None

        # Sparse Jacobian equations
        self.dfx_equ = None
        self.dfy_equ = None
        self.dgx_equ = None
        self.dgy_equ = None

        # Sets to store sparsity pattern
        self.sparsity_fx = list()
        self.sparsity_fy = list()
        self.sparsity_gx = list()
        self.sparsity_gy = list()

        # Dictionary with all the parameters
        self.params_dict = defaultdict(dict)

        # Dictionary with all the residuals for updating jacobian
        self.update_xy_dict = defaultdict(dict)

        # NOTE: To change!
        self.Tf = list()

    def add_to_f_g(self, eq_type_array, index, value):
        """
        Add value to a residual equation (f or g).
        :param eq_type_array: Residual array (f or g).
        :param index: Index within global system (needs nx offset for g)
        :param value: Value to add to the equation.
        :return:
        """
        eq_type_array[index - self.nx] += value

    def add_to_jacobian_init(self, jac_positions, jac_values, jac_equations, sparsity_set, row, col, value, equ):
        """
        Add a value to a Jacobian entry and record its sparsity pattern.
        :param jac_positions: list accumulating Jacobian positions
        :param jac_values: list accumulating Jacobian values
        :param sparsity_set: List of (row, col) pairs for sparsity tracking
        :param row: Jacobian row index
        :param col: Jacobian column index
        :param value: Value to insert or accumulate at (row, col)
        :return:
        """

        if (row, col) in jac_positions:
            jac_values[jac_positions.index((row, col))] += value
            jac_equations[jac_positions.index((row, col))] = jac_equations[jac_positions.index((row, col))] + equ
        else:
            jac_positions.append((row, col))
            jac_values.append(value)
            jac_equations.append(equ)
            sparsity_set.append((row, col))  # Store pattern

    def add_to_jacobian(self, jac_positions, jac_values, sparsity_set, row, col, value):
        """
        Add a value to a Jacobian entry and record its sparsity pattern.
        :param jac_positions: list accumulating Jacobian positions
        :param jac_values: list accumulating Jacobian values
        :param sparsity_set: List of (row, col) pairs for sparsity tracking
        :param row: Jacobian row index
        :param col: Jacobian column index
        :param value: Value to insert or accumulate at (row, col)
        :return:
        """

        if (row, col) in jac_positions:
            jac_values[jac_positions.index((row, col))] += value
        else:
            jac_positions.append((row, col))
            jac_values.append(value)
            sparsity_set.append((row, col))  # Store pattern

    def build_sparse_matrix_init(self, jac_positions, jac_values, jac_equations, sparsity_set, shape, jac_type):
        """
        Construct a sparse Jacobian matrix from dictionary and sparsity set
        :param jac_positions: list accumulating Jacobian positions
        :param jac_values: list accumulating Jacobian values
        :param sparsity_set: List of (row, col) pairs for sparsity tracking
        :param shape: Final matrix shape
        :param jac_type: Type of Jacobian ('dfx', 'dfy', 'dgx', or 'dgy')
        :return:
        """
        rows, cols = zip(*sparsity_set) if sparsity_set else ([], [])

        # Enable pretty/LaTeX printing
        sp.init_printing(use_latex=True)

        # Create a symbolic matrix
        jacobian_equations = zeros(shape[0],shape[1])

        if jac_type == 'dfx':
            values = [jac_values[jac_positions.index((r, c))] for r, c in sparsity_set]
            equations = [jac_equations[jac_positions.index((r, c))] for r, c in sparsity_set]
        if jac_type == 'dfy':
            values = [jac_values[jac_positions.index((r, c + self.nx))] for r, c in sparsity_set]
            equations = [jac_equations[jac_positions.index((r, c + self.nx))] for r, c in sparsity_set]
        if jac_type == 'dgx':
            values = [jac_values[jac_positions.index((r + self.nx, c))] for r, c in sparsity_set]
            equations = [jac_equations[jac_positions.index((r + self.nx, c))] for r, c in sparsity_set]
        if jac_type == 'dgy':
            values = [jac_values[jac_positions.index((r + self.nx, c + self.nx))] for r, c in sparsity_set]
            equations = [jac_equations[jac_positions.index((r + self.nx, c + self.nx))] for r, c in sparsity_set]

        for j, (row, col) in enumerate(sparsity_set):
            jacobian_equations[row, col] = equations[j]

        return coo_matrix((values, (rows, cols)), shape=shape), jacobian_equations

    def finalize_jacobians_init(self):
        """
        Finalize and build all Jacobian matrices as sparse matrices
        using the collected dictionary values and sparsity patterns.
        :return:
        """

        self.dfx, self.dfx_equ = self.build_sparse_matrix_init(self._dfx_jac_positions, self._dfx_jac_values, self._dfx_jac_equ,
                                            [(row, col) for row, col in self.sparsity_fx],
                                            (self.nx, self.nx), 'dfx')

        self.dfy, self.dfy_equ = self.build_sparse_matrix_init(self._dfy_jac_positions, self._dfy_jac_values, self._dfy_jac_equ,
                                            [(row, col - self.nx) for row, col in self.sparsity_fy],
                                            (self.nx, self.ny), 'dfy')

        self.dgx, self.dgx_equ = self.build_sparse_matrix_init(self._dgx_jac_positions, self._dgx_jac_values, self._dgx_jac_equ,
                                            [(row - self.nx, col) for row, col in self.sparsity_gx],
                                            (self.ny, self.nx), 'dgx')

        self.dgy, self.dgy_equ = self.build_sparse_matrix_init(self._dgy_jac_positions, self._dgy_jac_values, self._dgy_jac_equ,
                                            [(row - self.nx, col - self.nx) for row, col in self.sparsity_gy],
                                            (self.ny, self.ny), 'dgy')
        preview(self.dfx_equ, output = 'png', viewer = 'file', filename = 'jacobian_dfx.png')
        #with open('jacobian_dfx.png', 'wb') as f:º
         #  preview(self.dfx_equ, viewer='BytesIO', outputbuffer=f)

    def build_sparse_matrix(self, jac_positions, jac_values, sparsity_set, shape, jac_type):
        """
        Construct a sparse Jacobian matrix from dictionary and sparsity set
        :param jac_positions: list accumulating Jacobian positions
        :param jac_values: list accumulating Jacobian values
        :param sparsity_set: List of (row, col) pairs for sparsity tracking
        :param shape: Final matrix shape
        :param jac_type: Type of Jacobian ('dfx', 'dfy', 'dgx', or 'dgy')
        :return:
        """
        rows, cols = zip(*sparsity_set) if sparsity_set else ([], [])

        if jac_type == 'dfx':
            values = [jac_values[jac_positions.index((r, c))] for r, c in sparsity_set]

        if jac_type == 'dfy':
            values = [jac_values[jac_positions.index((r, c + self.nx))] for r, c in sparsity_set]

        if jac_type == 'dgx':
            values = [jac_values[jac_positions.index((r + self.nx, c))] for r, c in sparsity_set]

        if jac_type == 'dgy':
            values = [jac_values[jac_positions.index((r + self.nx, c + self.nx))] for r, c in sparsity_set]


        return coo_matrix((values, (rows, cols)), shape=shape)

    def finalize_jacobians(self):
        """
        Finalize and build all Jacobian matrices as sparse matrices
        using the collected dictionary values and sparsity patterns.
        :return:
        """

        self.dfx = self.build_sparse_matrix(self._dfx_jac_positions, self._dfx_jac_values,
                                            [(row, col) for row, col in self.sparsity_fx],
                                            (self.nx, self.nx), 'dfx')

        self.dfy = self.build_sparse_matrix(self._dfy_jac_positions, self._dfy_jac_values,
                                            [(row, col - self.nx) for row, col in self.sparsity_fy],
                                            (self.nx, self.ny), 'dfy')

        self.dgx = self.build_sparse_matrix(self._dgx_jac_positions, self._dgx_jac_values,
                                            [(row - self.nx, col) for row, col in self.sparsity_gx],
                                            (self.ny, self.nx), 'dgx')

        self.dgy = self.build_sparse_matrix(self._dgy_jac_positions, self._dgy_jac_values,
                                            [(row - self.nx, col - self.nx) for row, col in self.sparsity_gy],
                                            (self.ny, self.ny), 'dgy')


    def initilize_fg(self):
        """
        Initial setup of the residual system (f, g) and their Jacobians.
        This is typically called at the beginning of the simulation.
        :return:
        """
        self.concatenate()
        self.initialize_jacobian()
        self.finalize_jacobians_init()
        # demo
        logging.info(f"Jacobian computed, printing numeric and symbolic Jacobians" + '\n\n')
        logging.info(f"numeric dfx:" + '\n' + f"{self.dfx.toarray()}")
        logging.info(f"numeric dfy:" + '\n' + f"{self.dfy.toarray()}")
        logging.info(f"numeric dgx:" + '\n' + f"{self.dgx.toarray()}")
        logging.info(f"numeric dgy:" + '\n' + f"{self.dgy.toarray()}")
        logging.info(f"symbolic dfx:" + '\n' + f"{self.dfx_equ}")
        logging.info(f"symbolic dfy:" + '\n' + f"{self.dfy_equ}")
        logging.info(f"symbolic dgx:" + '\n' + f"{self.dgx_equ}")
        logging.info(f"symbolic dgy:" + '\n' + f"{self.dgy_equ}")

    def update_fg(self):
        """
        Recompute residuals and Jacobians (e.g., during simulation).
        Called during every iteration or step.
        :return:
        """
        self.concatenate()
        self.update_jacobian()
        self.finalize_jacobians()

    def concatenate(self):
        """
        Concatenate state and algebraic variables into one vector: [x; y].
        Useful for solving and passing into residual/Jacobian functions.
        :return:
        """
        self.xy = np.hstack((self.x, self.y))

    def finalize_tconst_matrix(self):
        """
        Build a sparse diagonal matrix from the Tf list.
        Typically used for time constant scaling in integration.
        :return:
        """
        self.Tf = diags(self.Tf)

    # Initialization
    def initialize_jacobian(self):
        """
        Initialize the Jacobian matrices and their sparsity patterns.
        This is typically called at the beginning of the simulation.
        :return:
        """
        # Reset the Jacobian lists and sparsity patterns
        self.f = np.zeros(self.nx)
        self.g = np.zeros(self.ny)

        self.sparsity_fx = list()
        self.sparsity_fy = list()
        self.sparsity_gx = list()
        self.sparsity_gy = list()

        self._dfx_jac_positions = list()
        self._dfy_jac_positions = list()
        self._dgx_jac_positions = list()
        self._dgy_jac_positions = list()

        self._dfx_jac_values = list()
        self._dfy_jac_values = list()
        self._dgx_jac_values = list()
        self._dgy_jac_values = list()

        for device in self.system.devices.values():
            if device.name != 'Bus':
                device.g_inputs_order = [[] for i in range(device.n)]
                device.f_inputs_order = [[] for i in range(device.n)]
                device.f_jac_inputs_order = [[] for i in range(device.n)]
                device.g_jac_inputs_order = [[] for i in range(device.n)]

                device.f_output_order = [[] for i in range(device.n)]
                device.g_output_order = [[] for i in range(device.n)]
                device.dfx_jac_output_order = [[] for i in range(device.n)]
                device.dfy_jac_output_order = [[] for i in range(device.n)]
                device.dgx_jac_output_order = [[] for i in range(device.n)]
                device.dgy_jac_output_order = [[] for i in range(device.n)]

                # Store input values in device
                self.get_input_values(device)

            # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template
            if device.name != 'Bus':
                ###f and g update
                # get local f and g info and values
                local_f_values, local_g_values, variables_names_for_ordering_f, variables_names_for_ordering_g = device.calc_f_g_functions()

                eq_type = 'f'
                pairs = self.assign_global_f_g_positions(device, local_f_values, variables_names_for_ordering_f,
                                                         device.f_output_order)
                for index, val in pairs:
                    self.add_to_f_g(self.f, index, val)

                eq_type = 'g'
                pairs = self.assign_global_f_g_positions(device, local_g_values, variables_names_for_ordering_g,
                                                         device.g_output_order)
                for index, val in pairs:
                    self.add_to_f_g(self.g, index, val)

                ### Jacobian update
                # get local jacobians info and values
                f_jacobians, g_jacobians, jacobian_info, jacobian_equations = device.calc_local_jacs()

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, f_jacobians, positions, equations, device.dfx_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dfx_jac_positions, self._dfx_jac_values, self._dfx_jac_equ, self.sparsity_fx, row, col, val, equ)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, f_jacobians, positions, equations, device.dfy_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dfy_jac_positions, self._dfy_jac_values, self._dfy_jac_equ, self.sparsity_fy, row, col, val, equ)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, g_jacobians, positions, equations, device.dgx_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dgx_jac_positions, self._dgx_jac_values, self._dgx_jac_equ, self.sparsity_gx, row, col, val, equ)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, g_jacobians, positions, equations, device.dgy_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dgy_jac_positions, self._dgy_jac_values, self._dgy_jac_equ, self.sparsity_gy, row, col, val, equ)

    def assign_global_f_g_positions(self, device, local_values, variables_names_for_ordering, outputs_order_list):
        """
        Assign global positions for f and g values.
        :param device: The device object containing the local values and output order
        :param local_values: Local values for f or
        :param variables_names_for_ordering: Names of the variables in the output order
        :param outputs_order_list: List to store the global output order
        :return: List of tuples containing the global index and corresponding value
        """

        # Initialize pairs list to store global indices and values
        pairs = []
        for i in range(device.n):

            for val, var_name in zip(local_values[i], variables_names_for_ordering):
                address = self.addresses_list[device.index][self.variables_list[device.index].index(var_name)][i]
                outputs_order_list[i].append(address)
                pairs.append((address, val))

        return pairs

    def assign_global_jac_positions(self, device, local_jacobian, positions, equations, outputs_order_triplets):
        """
        Assign global positions for Jacobian values.
        :param device: The device object containing the local Jacobian and output order
        :param local_jacobian: Local Jacobian values
        :param positions: List of tuples containing function and variable indices of local jacobian
        :param outputs_order_triplets: List to store the global output order
        :return: List of tuples containing global row, column, and value
        """
        # Initialize triplets list to store global row, column, and value
        triplets = []
        for i in range(device.n):
            for j, (func_index, var_index) in enumerate(positions):
                equation_str = sp.sympify(equations[j])
                val = local_jacobian[i][func_index][var_index]
                address_func = self.addresses_list[device.index][
                    self.variables_list[device.index].index(device.variables_list[func_index])][i]
                address_var = self.addresses_list[device.index][
                    self.variables_list[device.index].index(device.variables_list[var_index])][i]
                outputs_order_triplets[i].append((address_func, address_var))
                triplets.append((address_func, address_var, val, equation_str))

        return triplets

    def get_input_values(self, device):
        """
         Get the input values for the device.
        :param device: The device object for which to get the input values
        :return:
        """
        # Initialize input values lists for f and g
        values = self.xy

        self.build_input_values(values, device, device.f_args, device.f_input_values, device.f_inputs_order)
        self.build_input_values(values, device, device.g_args, device.g_input_values, device.g_inputs_order)
        self.build_input_values(values, device, device.f_jac_args, device.f_jac_input_values, device.f_jac_inputs_order)
        self.build_input_values(values, device, device.g_jac_args, device.g_jac_input_values, device.g_jac_inputs_order)

    def build_input_values(self, values, device, arguments_list, input_values_list, inputs_order_list):
        """
        Build input values for the device based on the provided arguments.
        :param values: The current state of the system
        :param device: The device object for which to build input values
        :param arguments_list: List of arguments for the device function
        :param input_values_list: List to store the input values
        :param inputs_order_list: List to store the order of inputs
        :return:
        """
        # Initialize input values lists for f and g
        for arg in arguments_list:
            for i in range(device.n):
                if arg in device.variables_list:
                    inputs_order_list[i].append(
                        self.addresses_list[device.index][self.variables_list[device.index].index(arg)][i])
                    input_values_list[i].append(
                        values[self.addresses_list[device.index][self.variables_list[device.index].index(arg)][i]])
                else:
                    inputs_order_list[i].append('param')
                    param = getattr(device, arg)
                    input_values_list[i].append(param.value[i])

    # Iterations
    def update_jacobian(self):
        """
        Update the Jacobian matrices and their sparsity patterns.
        This is typically called during the simulation iterations.
        :return:
        """
        # Reset the Jacobian dictionaries and sparsity patterns
        self._dfx_dict = {}
        self._dfy_dict = {}
        self._dgx_dict = {}
        self._dgy_dict = {}
        self.f = np.zeros(self.nx)
        self.g = np.zeros(self.ny)
        self.sparsity_fx = list()
        self.sparsity_fy = list()
        self.sparsity_gx = list()
        self.sparsity_gy = list()

        self._dfx_jac_positions = list()
        self._dfy_jac_positions = list()
        self._dgx_jac_positions = list()
        self._dgy_jac_positions = list()

        self._dfx_jac_values = list()
        self._dfy_jac_values = list()
        self._dgx_jac_values = list()
        self._dgy_jac_values = list()

        for device in self.system.devices.values():

            device.g_input_values = [[] for i in range(device.n)]
            device.f_input_values = [[] for i in range(device.n)]
            device.f_jac_input_values = [[] for i in range(device.n)]
            device.g_jac_input_values = [[] for i in range(device.n)]

            if device.name != 'Bus':
                self.get_fast_input_values(device)

            # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template
            if device.name != 'Bus':
                ###f and g update
                # get local f and g info and values
                local_f_values, local_g_values, variables_names_for_ordering_f, variables_names_for_ordering_g = device.calc_f_g_functions()

                eq_type = 'f'
                pairs = self.assign_fast_global_f_g_positions(device, local_f_values, device.f_output_order)
                for index, val in pairs:
                    self.add_to_f_g(self.f, index, val)

                eq_type = 'g'
                pairs = self.assign_fast_global_f_g_positions(device, local_g_values, device.g_output_order)
                for index, val in pairs:
                    self.add_to_f_g(self.g, index, val)

                ### Jacobian update
                # get local jacobians info and values
                f_jacobians, g_jacobians, jacobian_info, jacobian_equations = device.calc_local_jacs()

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, f_jacobians, positions,
                                                                 device.dfx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfx_jac_positions, self._dfx_jac_values, self.sparsity_fx, row, col, val)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, f_jacobians, positions,
                                                                 device.dfy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfy_jac_positions, self._dfy_jac_values, self.sparsity_fy, row, col, val)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, g_jacobians, positions,
                                                                 device.dgx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgx_jac_positions, self._dgx_jac_values, self.sparsity_gx, row, col, val)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, g_jacobians, positions,
                                                                 device.dgy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgy_jac_positions, self._dgy_jac_values, self.sparsity_gy, row, col, val)

    def assign_fast_global_f_g_positions(self, device, local_values, outputs_order_list):
        """
        Assign global positions for f and g values.
        :param device: The device object containing the local values and output order
        :param local_values: Local values for f or g
        :param outputs_order_list: List to store the global output order
        :return: List of tuples containing the global index and corresponding value
        """
        # Initialize pairs list to store global indices and values
        pairs = []
        for i in range(device.n):

            for address, val in zip(outputs_order_list[i], local_values[i]):
                pairs.append((address, val))

        return pairs

    def assign_fast_global_jac_positions(self, device, local_jacobian, positions, outputs_order_triplets):
        """
        Assign global positions for Jacobian values
        :param device: The device object containing the local Jacobian and output order
        :param local_jacobian: Local Jacobian value
        :param positions: List of tuples containing function and variable indices for local jacobian
        :param outputs_order_triplets: List to store the global output order
        :return: List of tuples containing global row, column, and value
        """
        # Initialize triplets list to store global row, column, and value
        triplets = []

        for i in range(device.n):

            for (func_index, var_index), (address_function, address_variable) in zip(positions,
                                                                                     outputs_order_triplets[i]):
                val = local_jacobian[i][func_index][var_index]
                address_func = address_function
                address_var = address_variable
                triplets.append((address_func, address_var, val))

        return triplets

    def get_fast_input_values(self, device):
        """
        Get the input values for the device.
        :param device: The device object for which to get the input values
        :return: f_input_values: List of input values for f.
                g_input_values: List of input values for g.
                f_jac_input_values: List of input values for f Jacobian.
                g_jac_input_values: List of input values for g Jacobian.
        """
        values = self.xy

        self.build_fast_input_values(values, device, device.f_args, device.f_input_values, device.f_inputs_order)
        self.build_fast_input_values(values, device, device.g_args, device.g_input_values, device.g_inputs_order)
        self.build_fast_input_values(values, device, device.f_jac_args, device.f_jac_input_values,
                                     device.f_jac_inputs_order)
        self.build_fast_input_values(values, device, device.g_jac_args, device.g_jac_input_values,
                                     device.g_jac_inputs_order)

    def build_fast_input_values(self, values, device, arguments_list, input_values_list, inputs_order_list):
        """
        Build input values for the device based on the provided arguments.
        :param values: The current state of the system
        :param device: The device object for which to build input value
        :param arguments_list: List of arguments for the device function
        :param input_values_list: List to store the input values
        :param inputs_order_list: List to store the order of inputs
        :return:
        """
        for j, arg in enumerate(arguments_list):

            for i in range(device.n):
                if inputs_order_list[i][j] != 'param':
                    input_values_list[i].append(values[inputs_order_list[i][j]])
                else:
                    param = getattr(device, arg)
                    input_values_list[i].append(param.value[i])
