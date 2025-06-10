# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List, Tuple
import numpy as np
import sympy as sym
import scipy.sparse as sp

from GridCalEngine.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Simulations.Rms.model.rms_model_store import RmsModelStore
from GridCalEngine.basic_structures import Vec


def build_dfx(jac_values, sparsity_set, n_rows: int, n_cols: int):
    """
    Construct a sparse Jacobian matrix from dictionary and sparsity set
    :param jac_values: list accumulating Jacobian values
    :param sparsity_set: List of (row, col) pairs for sparsity tracking
    :param n_rows: Number of rows
    :param n_cols: Number of columns
    :return:
    """
    for i in range(len(sparsity_set)):
        sparsity_set[i] = sparsity_set[i]

    if len(sparsity_set) != 0:
        rows, cols = zip(*sparsity_set)
    else:
        rows, cols = [], []

    return sp.coo_matrix((jac_values, (rows, cols)), shape=(n_rows, n_cols))


def build_dfy(jac_values, sparsity_set, n_rows: int, n_cols: int):
    """
    Construct a sparse Jacobian matrix from dictionary and sparsity set
    :param jac_values: list accumulating Jacobian values
    :param sparsity_set: List of (row, col) pairs for sparsity tracking
    :param n_rows: Number of rows
    :param n_cols: Number of columns
    :return:
    """

    for i in range(len(sparsity_set)):
        sparsity_set[i][1] = sparsity_set[i][1] - n_rows

    if len(sparsity_set) != 0:
        rows, cols = zip(*sparsity_set)
    else:
        rows, cols = [], []

    return sp.coo_matrix((jac_values, (rows, cols)), shape=(n_rows, n_cols))


def build_dgx(jac_values, sparsity_set, n_rows: int, n_cols: int):
    """
    Construct a sparse Jacobian matrix from dictionary and sparsity set
    :param jac_values: list accumulating Jacobian values
    :param sparsity_set: List of (row, col) pairs for sparsity tracking
    :param n_rows: Number of rows
    :param n_cols: Number of columns
    :return:
    """
    for i in range(len(sparsity_set)):
        sparsity_set[i][0] = sparsity_set[i][0] - n_cols

    if len(sparsity_set) != 0:
        rows, cols = zip(*sparsity_set)
    else:
        rows, cols = [], []

    return sp.coo_matrix((jac_values, (rows, cols)), shape=(n_rows, n_cols))


def build_dgy(jac_values, sparsity_set, n_rows: int, n_cols: int, offset_rows, offset_cols):
    """
    Construct a sparse Jacobian matrix from dictionary and sparsity set
    :param jac_values: list accumulating Jacobian values
    :param sparsity_set: List of (row, col) pairs for sparsity tracking
    :param n_rows: Number of rows
    :param n_cols: Number of columns
    :return:
    """

    for i in range(len(sparsity_set)):
        sparsity_set[i][0] = sparsity_set[i][0] - offset_rows
        sparsity_set[i][1] = sparsity_set[i][1] - offset_cols

    if len(sparsity_set) != 0:
        rows, cols = zip(*sparsity_set)
    else:
        rows, cols = [], []

    return sp.coo_matrix((jac_values, (rows, cols)), shape=(n_rows, n_cols))


def compile_rms_models(grid: MultiCircuit) -> Tuple[List[RmsModelStore], Vec, Vec, Vec, int, int]:
    """

    :param grid:
    :return:
    """
    models: List[RmsModelStore] = list()
    n_algeb = 0
    n_stat = 0

    already_compiled_dict = dict()

    for lst in [grid.buses,
                grid.get_injection_devices_iter(),
                grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)]:

        for elm in lst:

            # obtain the used model from the device of the DB
            model = elm.rms_model.model

            # See if the dynamic model was already compiled
            c_model = already_compiled_dict.get(model.idtag, None)

            if c_model is None:
                # if it wasn't compiled, compile it!
                n_algeb += len(model.algeb_var)
                n_stat += len(model.stat_var)
                c_model = RmsModelStore(dynamic_model=elm.rms_model.model, grid_id=grid.idtag)

                # store reference for later
                already_compiled_dict[model.idtag] = c_model

            # add the compiled model used to the list
            models.append(c_model)

    # Compile initial values
    x0 = np.empty(n_stat)
    y0 = np.empty(n_algeb)
    Tf = np.ones(n_stat)

    a_stat = 0
    b_stat = 0
    a_algeb = 0
    b_algeb = 0
    for c_model in models:
        b_stat += len(c_model.x0)
        x0[a_stat:b_stat] = c_model.x0
        Tf[a_stat:b_stat] = c_model.t_const0
        a_stat = b_stat

        b_algeb += len(c_model.y0)
        y0[a_algeb:b_algeb] = c_model.y0
        a_algeb = b_algeb

    return models, x0, y0, Tf, n_stat, n_algeb


class RmsProblem:
    """
    DAE (Differential-Algebraic Equation) class to store and manage.

    Responsibilities:
        - Store state and algebraic variables (x, y)
        - Store Jacobian matrices
        - Store residual equations
        - Store sparsity patterns
    """

    def __init__(self, grid: MultiCircuit):
        """
        DAE class constructor
        Initialize DAE object with required containers and defaults.
        :param system: The simulation system object to which this DAE is tied
        """

        # self.system: DynamicSystemStore = system
        self.models_list, self.x, self.y, self.Tf, self.nx, self.ny = compile_rms_models(grid=grid)

        # lists to store initial values and results of every step of the simulation
        self.xy = None

        # List to store variables internal variables of the system (used to analyze simulation data)
        self.internal_variables_list = list()

        # List to store all variables (used when updating f, g, and jacobian values)
        self.variables_list = list()

        # List to store all addresses (used when updating f, g, and jacobian values)
        self.addresses_list = list()

        # number of state variables (internals)
        # self.nx = 0
        # number of algebraic variables (internals)
        # self.ny = 0

        # number of non zero entries in jacobian
        self.ndfx = 0
        self.ndfy = 0
        self.ndgx = 0
        self.ndgy = 0

        # lists to store f ang g functions values (residuals)
        self.f = np.zeros(self.nx)
        self.g = np.zeros(self.ny)

        # Lists to accumulate Jacobian positions, values and equations

        self._dfx_jac_positions = np.zeros(self.ndfx, dtype=object)
        self._dfy_jac_positions = np.zeros(self.ndfy, dtype=object)
        self._dgx_jac_positions = np.zeros(self.ndgx, dtype=object)
        self._dgy_jac_positions = np.zeros(55, dtype=object)  # TODO: investigate 2-pass sizing

        self._dfx_jac_values = np.zeros(self.ndfx)
        self._dfy_jac_values = np.zeros(self.ndfy)
        self._dgx_jac_values = np.zeros(self.ndgx)
        self._dgy_jac_values = np.zeros(55)  # TODO: investigate 2-pass sizing

        self._dfx_jac_equ = np.zeros(self.ndfx, dtype=object)
        self._dfy_jac_equ = np.zeros(self.ndfy, dtype=object)
        self._dgx_jac_equ = np.zeros(self.ndgx, dtype=object)
        self._dgy_jac_equ = np.zeros(55, dtype=object)  # TODO: investigate 2-pass sizing

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
        self.sparsity_fx = np.zeros(self.ndfx, dtype=object)
        self.sparsity_fy = np.zeros(self.ndfy, dtype=object)
        self.sparsity_gx = np.zeros(self.ndgx, dtype=object)
        self.sparsity_gy = np.zeros(55, dtype=object)

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
        # logging.info(f"Jacobian computed, printing numeric and symbolic Jacobians" + '\n\n')
        # logging.info(f"numeric dfx:" + '\n' + f"{self.dfx.toarray()}")
        # logging.info(f"numeric dfy:" + '\n' + f"{self.dfy.toarray()}")
        # logging.info(f"numeric dgx:" + '\n' + f"{self.dgx.toarray()}")
        # logging.info(f"numeric dgy:" + '\n' + f"{self.dgy.toarray()}")
        # with open('jacobian_dfx.png', 'wb') as f:
        #     preview(self.dfx_equ, viewer='BytesIO', outputbuffer=f)
        # with open('jacobian_dfy.png', 'wb') as f:
        #     preview(self.dfy_equ, viewer='BytesIO', outputbuffer=f)
        # with open('jacobian_dgx.png', 'wb') as f:
        #     preview(self.dgx_equ, viewer='BytesIO', outputbuffer=f)
        # with open('jacobian_dgy.png', 'wb') as f:
        #     preview(self.dgy_equ, viewer='BytesIO', outputbuffer=f)

    def concatenate(self):
        """
        Concatenate state and algebraic variables into one vector: [x; y].
        Useful for solving and passing into residual/Jacobian functions.
        :return:
        """
        self.xy = np.hstack((self.x, self.y))

    def initialize_jacobian(self):
        """
        Initialize the Jacobian matrices and their sparsity patterns.
        This is typically called at the beginning of the simulation.
        :return:
        """
        # Reset the Jacobian lists and sparsity patterns
        self.f = np.zeros(self.nx)
        self.g = np.zeros(self.ny)

        self.sparsity_fx = np.zeros(self.ndfx, dtype=object)
        self.sparsity_fy = np.zeros(self.ndfy, dtype=object)
        self.sparsity_gx = np.zeros(self.ndgx, dtype=object)
        self.sparsity_gy = np.zeros(55, dtype=object)

        self._dfx_jac_positions = np.zeros(self.ndfx, dtype=object)
        self._dfy_jac_positions = np.zeros(self.ndfy, dtype=object)
        self._dgx_jac_positions = np.zeros(self.ndgx, dtype=object)
        self._dgy_jac_positions = np.zeros(55, dtype=object)

        self._dfx_jac_values = np.zeros(self.ndfx)
        self._dfy_jac_values = np.zeros(self.ndfy)
        self._dgx_jac_values = np.zeros(self.ndgx)
        self._dgy_jac_values = np.zeros(55)

        self._dfx_jac_equ = np.zeros(self.ndfx, dtype=object)
        self._dfy_jac_equ = np.zeros(self.ndfy, dtype=object)
        self._dgx_jac_equ = np.zeros(self.ndgx, dtype=object)
        self._dgy_jac_equ = np.zeros(55, dtype=object)

        for device in self.models_list:
            if device.name != 'Bus':
                # Initialize lists to store the addresses of the variables in the order of the input of the functions.
                device.f_inputs_order = np.zeros((device.n, len(device.f_args)), dtype=object)
                device.g_inputs_order = np.zeros((device.n, len(device.g_args)), dtype=object)

                # Initialize lists to store the addresses of the variables in the order of the input of the functions.
                device.f_jac_inputs_order = np.zeros((device.n, len(device.f_jac_arguments)), dtype=object)
                device.g_jac_inputs_order = np.zeros((device.n, len(device.g_jac_arguments)), dtype=object)

                # Store input values in device

                device.f_input_values = np.zeros((device.n, len(device.f_args)), dtype=object)
                device.g_input_values = np.zeros((device.n, len(device.g_args)), dtype=object)
                device.f_jac_input_values = np.zeros((device.n, len(device.f_jac_arguments)), dtype=object)
                device.g_jac_input_values = np.zeros((device.n, len(device.g_jac_arguments)), dtype=object)

                self.get_input_values(device)

            # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template
            if device.name != 'Bus':
                ###f and g update
                # get local f and g info and values
                local_f_values, local_g_values, variables_names_for_ordering_f, variables_names_for_ordering_g = device.calc_f_g_functions()

                # Initialize lists to store addresses of the variables in the order og the output of the functions.
                device.f_output_order = np.zeros_like(local_f_values, dtype=object)
                device.g_output_order = np.zeros_like(local_g_values, dtype=object)

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
                jacobian, jacobian_info, jacobian_equations = device.calc_local_jacs()

                # Initialize lists to store addresses of the variables in the order of the output of the functions.
                device.dfx_jac_output_order = np.zeros((device.n, len(jacobian_info['dfx'])), dtype=object)
                device.dfy_jac_output_order = np.zeros((device.n, len(jacobian_info['dfy'])), dtype=object)
                device.dgx_jac_output_order = np.zeros((device.n, len(jacobian_info['dgx'])), dtype=object)
                device.dgy_jac_output_order = np.zeros((device.n, len(jacobian_info['dgy'])), dtype=object)

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, jacobian, positions, equations,
                                                            device.dfx_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dfx_jac_positions, self._dfx_jac_values, self._dfx_jac_equ,
                                              self.sparsity_fx, row, col, val, equ)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, jacobian, positions, equations,
                                                            device.dfy_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dfy_jac_positions, self._dfy_jac_values, self._dfy_jac_equ,
                                              self.sparsity_fy, row, col, val, equ)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, jacobian, positions, equations,
                                                            device.dgx_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dgx_jac_positions, self._dgx_jac_values, self._dgx_jac_equ,
                                              self.sparsity_gx, row, col, val, equ)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                equations = jacobian_equations[jac_type]
                triplets = self.assign_global_jac_positions(device, jacobian, positions, equations,
                                                            device.dgy_jac_output_order)
                for row, col, val, equ in triplets:
                    self.add_to_jacobian_init(self._dgy_jac_positions, self._dgy_jac_values, self._dgy_jac_equ,
                                              self.sparsity_gy, row, col, val, equ)

    def get_input_values(self, device: RmsModelStore):
        """
         Get the input values for the device.
        :param device: The device object for which to get the input values
        :return:
        """
        # Initialize input values lists for f and g
        self.build_input_values(self.xy, device, device.f_args, device.f_input_values, device.f_inputs_order)
        self.build_input_values(self.xy, device, device.g_args, device.g_input_values, device.g_inputs_order)
        self.build_input_values(self.xy, device, device.f_jac_arguments, device.f_jac_input_values,
                                device.f_jac_inputs_order)
        self.build_input_values(self.xy, device, device.g_jac_arguments, device.g_jac_input_values,
                                device.g_jac_inputs_order)

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

        for j, arg in enumerate(arguments_list):
            for i in range(device.n):
                if arg in device.variables_list:
                    inputs_order_list[i][j] = \
                        self.addresses_list[device.index][self.variables_list[device.index].index(arg)][i]
                    input_values_list[i][j] = values[
                        self.addresses_list[device.index][self.variables_list[device.index].index(arg)][i]]

                else:
                    inputs_order_list[i][j] = 'param'
                    param = getattr(device, arg)
                    input_values_list[i][j] = param.value[i]

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
            for j, (val, var_name) in enumerate(zip(local_values[i], variables_names_for_ordering)):
                address = self.addresses_list[device.index][self.variables_list[device.index].index(var_name)][i]
                outputs_order_list[i][j] = address
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

        triplets = []

        for i in range(device.n):

            for j, (func_index, var_index) in enumerate(positions):
                equation_str = sym.sympify(equations[j])
                val = local_jacobian[i][func_index][var_index]
                address_func = \
                    self.addresses_list[device.index][
                        self.variables_list[device.index].index(device.eqs_list[func_index])][
                        i]
                address_var = \
                    self.addresses_list[device.index][
                        self.variables_list[device.index].index(device.vars_list[var_index])][
                        i]
                outputs_order_triplets[i][j] = (address_func, address_var)
                triplets.append((address_func, address_var, val, equation_str))

        return triplets

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
        in_positions = [row, col] in jac_positions.tolist()
        if in_positions:
            index = jac_positions.tolist().index([row, col])
            jac_values[index] += value
            jac_equations[index] = jac_equations[index] + equ
        else:
            index = np.where(jac_positions == 0)[0]
            jac_positions[index[0]] = [row, col]
            jac_values[index[0]] = value
            jac_equations[index[0]] = equ
            sparsity_set[index[0]] = [row, col]  # Store pattern

    def finalize_jacobians_init(self):
        """
        Finalize and build all Jacobian matrices as sparse matrices
        using the collected dictionary values and sparsity patterns.
        :return:
        """

        self.dfx, self.dfx_equ = self.build_sparse_matrix_init(self._dfx_jac_positions, self._dfx_jac_values,
                                                               self._dfx_jac_equ, self.sparsity_fx,
                                                               (self.nx, self.nx), 'dfx')

        self.dfy, self.dfy_equ = self.build_sparse_matrix_init(self._dfy_jac_positions, self._dfy_jac_values,
                                                               self._dfy_jac_equ,
                                                               self.sparsity_fy,
                                                               (self.nx, self.ny), 'dfy')

        self.dgx, self.dgx_equ = self.build_sparse_matrix_init(self._dgx_jac_positions, self._dgx_jac_values,
                                                               self._dgx_jac_equ,
                                                               self.sparsity_gx,
                                                               (self.ny, self.nx), 'dgx')

        self.dgy, self.dgy_equ = self.build_sparse_matrix_init(self._dgy_jac_positions, self._dgy_jac_values,
                                                               self._dgy_jac_equ,
                                                               self.sparsity_gy,
                                                               (self.ny, self.ny), 'dgy')

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

        # Enable pretty/LaTeX printing
        sym.init_printing(use_latex=True)

        # Create a symbolic matrix
        jacobian_equations = sym.zeros(shape[0], shape[1])

        if jac_type == 'dfx':
            values = jac_values.tolist()
            equations = jac_equations.tolist()
            for i in range(len(sparsity_set)):
                sparsity_set[i] = sparsity_set[i]
            if len(sparsity_set) != 0:
                rows, cols = zip(*sparsity_set)
            else:
                rows, cols = [], []


        elif jac_type == 'dfy':
            values = jac_values.tolist()
            equations = jac_equations.tolist()
            for i in range(len(sparsity_set)):
                sparsity_set[i][1] = sparsity_set[i][1] - self.nx
            if len(sparsity_set) != 0:
                rows, cols = zip(*sparsity_set)
            else:
                rows, cols = [], []

        elif jac_type == 'dgx':
            values = jac_values.tolist()
            equations = jac_equations.tolist()
            for i in range(len(sparsity_set)):
                sparsity_set[i][0] = sparsity_set[i][0] - self.nx
            if len(sparsity_set) != 0:
                rows, cols = zip(*sparsity_set)
            else:
                rows, cols = [], []

        elif jac_type == 'dgy':
            values = jac_values.tolist()
            equations = jac_equations.tolist()
            for i in range(len(sparsity_set)):
                sparsity_set[i][0] = sparsity_set[i][0] - self.nx
                sparsity_set[i][1] = sparsity_set[i][1] - self.nx
            if len(sparsity_set) != 0:
                rows, cols = zip(*sparsity_set)
            else:
                rows, cols = [], []

        else:
            raise ValueError(f"jac_type is not expected: {jac_type}")

        for j, (row, col) in enumerate(sparsity_set):
            jacobian_equations[row, col] = equations[j]

        return sp.coo_matrix((values, (rows, cols)), shape=shape), jacobian_equations

    def update_fg(self):
        """
        Recompute residuals and Jacobians (e.g., during simulation).
        Called during every iteration or step.
        :return:
        """
        self.concatenate()
        self.update_jacobian()
        self.finalize_jacobians()

    def update_jacobian(self):
        """
        Update the Jacobian matrices and their sparsity patterns.
        This is typically called during the simulation iterations.
        :return:
        """
        # Reset the Jacobian info

        self.f = np.zeros(self.nx)
        self.g = np.zeros(self.ny)

        self.sparsity_fx = np.zeros(self.ndfx, dtype=object)
        self.sparsity_fy = np.zeros(self.ndfy, dtype=object)
        self.sparsity_gx = np.zeros(self.ndgx, dtype=object)
        self.sparsity_gy = np.zeros(55, dtype=object)

        self._dfx_jac_positions = np.zeros(self.ndfx, dtype=object)
        self._dfy_jac_positions = np.zeros(self.ndfy, dtype=object)
        self._dgx_jac_positions = np.zeros(self.ndgx, dtype=object)
        self._dgy_jac_positions = np.zeros(55, dtype=object)

        self._dfx_jac_values = np.zeros(self.ndfx)
        self._dfy_jac_values = np.zeros(self.ndfy)
        self._dgx_jac_values = np.zeros(self.ndgx)
        self._dgy_jac_values = np.zeros(55)

        for device in self.models_list:

            if device.name != 'Bus':

                device.f_input_values = np.zeros((device.n, len(device.f_args)), dtype=object)
                device.g_input_values = np.zeros((device.n, len(device.g_args)), dtype=object)
                device.f_jac_input_values = np.zeros((device.n, len(device.f_jac_arguments)), dtype=object)
                device.g_jac_input_values = np.zeros((device.n, len(device.g_jac_arguments)), dtype=object)

                self.get_fast_input_values(device)

                # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template

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
                jacobian, jacobian_info, jacobian_equations = device.calc_local_jacs()

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, jacobian, positions,
                                                                 device.dfx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfx_jac_positions, self._dfx_jac_values, self.sparsity_fx, row, col, val)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, jacobian, positions,
                                                                 device.dfy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfy_jac_positions, self._dfy_jac_values, self.sparsity_fy, row, col, val)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, jacobian, positions,
                                                                 device.dgx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgx_jac_positions, self._dgx_jac_values, self.sparsity_gx, row, col, val)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, jacobian, positions,
                                                                 device.dgy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgy_jac_positions, self._dgy_jac_values, self.sparsity_gy, row, col, val)

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
                    input_values_list[i][j] = values[inputs_order_list[i][j]]
                else:
                    param = getattr(device, arg)
                    input_values_list[i][j] = param.value[i]

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
        in_positions = [row, col] in jac_positions.tolist()
        if in_positions:
            index = jac_positions.tolist().index([row, col])
            jac_values[index] += value
        else:
            index = np.where(jac_positions == 0)[0]
            jac_positions[index[0]] = [row, col]
            jac_values[index[0]] = value
            sparsity_set[index[0]] = [row, col]  # Store pattern

    def finalize_jacobians(self):
        """
        Finalize and build all Jacobian matrices as sparse matrices
        using the collected dictionary values and sparsity patterns.
        :return:
        """

        self.dfx = build_dfx(self._dfx_jac_values, self.sparsity_fx, self.nx, self.nx)
        self.dfy = build_dfy(self._dfy_jac_values, self.sparsity_fy, self.nx, self.ny)
        self.dgx = build_dgx(self._dgx_jac_values, self.sparsity_gx, self.ny, self.nx)
        self.dgy = build_dgy(self._dgy_jac_values, self.sparsity_gy, self.ny, self.ny, self.nx, self.nx)

    def get_tconst_matrix(self):
        """
        Build a sparse diagonal matrix from the Tf list.
        Typically used for time constant scaling in integration.
        :return:
        """
        return sp.diags(self.Tf)
