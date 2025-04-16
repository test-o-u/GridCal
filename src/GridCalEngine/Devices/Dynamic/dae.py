import pdb

import numpy as np
import config
from scipy.sparse import coo_matrix, diags
from collections import defaultdict


class DAE:
    """
    DAE class to store numerical paramter, state and algebraic variable, jacobian and residual values.
    """

    def __init__(self, system):

        self.system = system

        # dictionary to store addresses
        self.addresses_dict = defaultdict(dict)

        self.nx = 0
        self.ny = 0

        self.x = config.DAE_X0
        self.y = config.DAE_Y0
        self.xy_unique = None
        self.f = np.zeros(2)
        self.g = np.zeros(13)

        # Dictionaries to accumulate Jacobian values
        self._dfx_dict = {}
        self._dfy_dict = {}
        self._dgx_dict = {}
        self._dgy_dict = {}

        # Sparse Jacobian values
        self.dfx = None
        self.dfy = None
        self.dgx = None
        self.dgy = None

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
        self.Tf = []

    def add_to_f_g(self, eq_type_array, index, value):
        eq_type_array[index - self.nx] += value

    def add_to_jacobian(self, jac_dict, sparsity_set, row, col, value):
        """
        Accumulate values and track sparsity pattern.
        """
        if (row, col) in jac_dict:
            jac_dict[(row, col)] += value
        else:
            jac_dict[(row, col)] = value  # First assignment
            sparsity_set.append((row, col))  # Store pattern

    def build_sparse_matrix(self, jac_dict, sparsity_set, shape, jac_type):
        """
        Convert accumulated values into a sparse matrix using the precomputed pattern.
        """

        rows, cols = zip(*sparsity_set) if sparsity_set else ([], [])
        if jac_type == 'dfx':
            values = [jac_dict.get((r, c), 0) for r, c in sparsity_set]
        if jac_type == 'dfy':
            values = [jac_dict.get((r, c + self.nx), 0) for r, c in sparsity_set]
        if jac_type == 'dgx':
            values = [jac_dict.get((r + self.nx, c), 0) for r, c in sparsity_set]
        if jac_type == 'dgy':
            values = [jac_dict.get((r + self.nx, c + self.nx), 0) for r, c in sparsity_set]

        return coo_matrix((values, (rows, cols)), shape=shape)

    def finalize_jacobians(self):
        """
        Builds all Jacobian matrices from stored triplets and sparsity patterns.
        """

        self.dfx = self.build_sparse_matrix(self._dfx_dict,
                                            [(row, col) for row, col in self.sparsity_fx],
                                            (self.nx, self.nx), 'dfx')

        self.dfy = self.build_sparse_matrix(self._dfy_dict,
                                            [(row, col - self.nx) for row, col in self.sparsity_fy],
                                            (self.nx, self.ny), 'dfy')

        self.dgx = self.build_sparse_matrix(self._dgx_dict,
                                            [(row - self.nx, col) for row, col in self.sparsity_gx],
                                            (self.ny, self.nx), 'dgx')

        self.dgy = self.build_sparse_matrix(self._dgy_dict,
                                            [(row - self.nx, col - self.nx) for row, col in self.sparsity_gy],
                                            (self.ny, self.ny), 'dgy')

    def initilize_fg(self):
        self.concatenate()
        self.initialize_jacobian()
        self.finalize_jacobians()

    def update_fg(self):
        self.concatenate()
        self.update_jacobian()
        self.finalize_jacobians()

    def concatenate(self):
        self.xy_unique = np.hstack((self.x, self.y))

    def finalize_tconst_matrix(self):
        self.Tf = diags(self.Tf)


    # Initialization

    def initialize_jacobian(self):
        self._dfx_dict = {}
        self._dfy_dict = {}
        self._dgx_dict = {}
        self._dgy_dict = {}
        self.f = np.zeros(2)
        self.g = np.zeros(13)
        self.sparsity_fx = list()
        self.sparsity_fy = list()
        self.sparsity_gx = list()
        self.sparsity_gy = list()

        for device in self.system.devices.values():
            if device.name != 'Bus':
                f_input_values, g_input_values, f_jac_input_values, g_jac_input_values = self.get_input_values(device)

            # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template
            if device.name != 'Bus':
                # get variable addresses
                var_addresses = self.addresses_dict[device.name]

                ###f and g update
                # get local f and g info and values
                local_f_values, local_g_values, variables_names_for_ordering_f, variables_names_for_ordering_g = device.calc_f_g_functions(
                    f_input_values, g_input_values)

                eq_type = 'f'
                pairs = self.assign_global_f_g_positions(device, local_f_values, variables_names_for_ordering_f, var_addresses, device.f_output_order)
                for index, val in pairs:
                    self.add_to_f_g(self.f, index, val)

                eq_type = 'g'
                pairs = self.assign_global_f_g_positions(device, local_g_values, variables_names_for_ordering_g, var_addresses, device.g_output_order)
                for index, val in pairs:
                    self.add_to_f_g(self.g, index, val)

                ### Jacobian update
                # get local jacobians info and values
                f_jacobians, g_jacobians, jacobian_info = device.calc_local_jacs(f_jac_input_values, g_jac_input_values)

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_global_jac_positions(device, f_jacobians, positions, var_addresses, device.dfx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfx_dict, self.sparsity_fx, row, col, val)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_global_jac_positions(device, f_jacobians, positions, var_addresses, device.dfy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfy_dict, self.sparsity_fy, row, col, val)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_global_jac_positions(device, g_jacobians, positions, var_addresses, device.dgx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgx_dict, self.sparsity_gx, row, col, val)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_global_jac_positions(device, g_jacobians, positions, var_addresses, device.dgy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgy_dict, self.sparsity_gy, row, col, val)

    def assign_global_f_g_positions(self, device, local_values, variables_names_for_ordering, var_addresses, outputs_order_list):
        pairs = []
        for i in range(device.n):
            for val, var_name in zip(local_values[i], variables_names_for_ordering):
                address = var_addresses[var_name][i]
                outputs_order_list.append(address)
                pairs.append((address, val))
        return pairs

    def assign_global_jac_positions(self, device, local_jacobian, positions, var_addresses, outputs_order_triplets):
        triplets = []
        for i in range(device.n):
            for j, (func_index, var_index) in enumerate(positions):
                val = local_jacobian[i][func_index][var_index]
                address_func = var_addresses[device.vars_index[func_index]][i]
                address_var = var_addresses[device.vars_index[var_index]][i]
                outputs_order_triplets.append((address_func, address_var))
                triplets.append((address_func, address_var, val))
        return triplets

    def get_input_values(self, device):

        g_input_values = [[] for i in range(device.n)]
        f_input_values = [[] for i in range(device.n)]
        f_jac_input_values = [[] for i in range(device.n)]
        g_jac_input_values = [[] for i in range(device.n)]

        generated_code = device.import_generated_code()

        values = self.xy_unique

        f_arguments = generated_code.f_args
        g_arguments = generated_code.g_args

        f_jac_arguments = generated_code.f_jac_args
        g_jac_arguments = generated_code.g_jac_args

        self.build_input_values(values, device, f_arguments, f_input_values, device.f_inputs_order)
        self.build_input_values(values, device, g_arguments, g_input_values, device.g_inputs_order)
        self.build_input_values(values, device, f_jac_arguments, f_jac_input_values, device.f_jac_inputs_order)
        self.build_input_values(values, device, g_jac_arguments, g_jac_input_values, device.g_jac_inputs_order)

        return f_input_values, g_input_values, f_jac_input_values, g_jac_input_values


    def build_input_values(self, values, device, arguments_list, input_values_list, inputs_order_list ):
        for arg in arguments_list:
            for i in range(device.n):
                if arg in device.variables_list:
                    inputs_order_list.append(self.addresses_dict[device.name][arg])
                    input_values_list[i].append(values[self.addresses_dict[device.name][arg]][i])
                else:
                    inputs_order_list.append('param')
                    param = getattr(device, arg)
                    input_values_list[i].append(param.value[i])

    # Iterations

    def update_jacobian(self):
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

        for device in self.system.devices.values():
            if device.name != 'Bus':
                f_input_values, g_input_values, f_jac_input_values, g_jac_input_values = self.get_fast_input_values(device)

            # Get the function type and var type info and the local jacobians using the calc_local_jacs function defined in dynamic_model_template
            if device.name != 'Bus':
                # get variable addresses
                var_addresses = self.addresses_dict[device.name]

                ###f and g update
                # get local f and g info and values
                local_f_values, local_g_values, variables_names_for_ordering_f, variables_names_for_ordering_g = device.calc_f_g_functions(
                    f_input_values, g_input_values)

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
                f_jacobians, g_jacobians, jacobian_info = device.calc_local_jacs(f_jac_input_values, g_jac_input_values)

                # calc dfx
                jac_type = 'dfx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, f_jacobians, positions, device.dfx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfx_dict, self.sparsity_fx, row, col, val)

                # calc dfy
                jac_type = 'dfy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, f_jacobians, positions, device.dfy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dfy_dict, self.sparsity_fy, row, col, val)

                # calc dgx
                jac_type = 'dgx'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, g_jacobians, positions, device.dgx_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgx_dict, self.sparsity_gx, row, col, val)

                # calc dgy
                jac_type = 'dgy'
                positions = jacobian_info[jac_type]
                triplets = self.assign_fast_global_jac_positions(device, g_jacobians, positions, device.dgy_jac_output_order)
                for row, col, val in triplets:
                    self.add_to_jacobian(self._dgy_dict, self.sparsity_gy, row, col, val)

    def assign_fast_global_f_g_positions(self, device, local_values, outputs_order_list):
        pairs = []
        for i in range(device.n):
            for val, address in zip(local_values[i], outputs_order_list):
                pairs.append((address, val))

        return pairs

    def assign_fast_global_jac_positions(self, device, local_jacobian, positions, outputs_order_triplets):
        triplets = []
        for i in range(device.n):
            for (func_index, var_index), (address_function, address_variable) in zip(positions, outputs_order_triplets):
                val = local_jacobian[i][func_index][var_index]
                address_func = address_function
                address_var = address_variable
                triplets.append((address_func, address_var, val))
        return triplets

    def get_fast_input_values(self, device):

        g_input_values = [[] for i in range(device.n)]
        f_input_values = [[] for i in range(device.n)]
        f_jac_input_values = [[] for i in range(device.n)]
        g_jac_input_values = [[] for i in range(device.n)]

        generated_code = device.import_generated_code()

        values = self.xy_unique

        f_arguments = generated_code.f_args
        g_arguments = generated_code.g_args

        f_jac_arguments = generated_code.f_jac_args
        g_jac_arguments = generated_code.g_jac_args

        self.build_fast_input_values(values, device, f_arguments, f_input_values, device.f_inputs_order)
        self.build_fast_input_values(values, device, g_arguments, g_input_values, device.g_inputs_order)
        self.build_fast_input_values(values, device, f_jac_arguments, f_jac_input_values, device.f_jac_inputs_order)
        self.build_fast_input_values(values, device, g_jac_arguments, g_jac_input_values, device.g_jac_inputs_order)

        return f_input_values, g_input_values, f_jac_input_values, g_jac_input_values


    def build_fast_input_values(self, values, device, arguments_list, input_values_list, inputs_order_list ):
        for j, arg in enumerate(arguments_list):
            for i in range(device.n):
                if inputs_order_list[j] != 'param':
                    input_values_list[i].append(values[inputs_order_list[j]][i])
                else:
                    param = getattr(device, arg)
                    input_values_list[i].append(param.value[i])