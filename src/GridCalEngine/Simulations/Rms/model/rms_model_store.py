# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import datetime
import os
import inspect
import pdb
import pprint
import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify

from GridCalEngine.Devices.Dynamic.dynamic_model import DynamicModel
from GridCalEngine.IO.file_system import (get_create_dynamics_model_folder, load_file_as_module,
                                          load_function_from_module)


class RmsModelStore:

    def __init__(self, dynamic_model: DynamicModel, grid_id: str):
        """

        :param dynamic_model:
        :param grid_id:
        """
        self.name = dynamic_model.name

        self.idtag = dynamic_model.idtag

        # # Device index for ordering devices in the system (used to relate variables with their addresses)
        # self.index = device_index

        # Set address function
        self.n = 1  # index for the number of components corresponding to this model

        # Symbolic processing engine utils
        self.stats = list()
        self.algebs = list()

        # to build adressing system
        self.variables = list()

        # dictionary containing index of the variable as key and symbol of the variable as value
        self.vars_index = np.empty(dynamic_model.get_var_num(), dtype=object)

        # Lists of internal and external variables
        self.internal_vars = list()
        self.output_vars = list()
        self.input_vars = list()

        # list containing all the symbols of the variables in the model (used in f, g, and jacobian calculation)
        self.variables_list = list()  # list of all the variables (including external)
        self.state_eqs = list()
        self.state_vars = list()
        self.algeb_eqs = list()
        self.algeb_vars = list()
        self.input_state_eqs = list()
        self.input_algeb_eqs = list()
        self.eqs_list = list()  # list of all the variables with an equation
        self.vars_list = list()  # list of all the variables
        self.real_state_vars = list()
        self.real_algeb_vars = list()

        # Dictionary with parameters
        self.num_params_dict = dynamic_model.num_dyn_param

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
        Symbolic vars
        """

        # Symbolic Equations
        self.f_args = list()
        self.g_args = list()
        self.f_list = list()
        self.g_list = list()
        self.f_matrix = ()
        self.g_matrix = ()
        self.lambda_equations = dict()
        self.variables_names_for_ordering = dict()

        # Jacobians
        self.f_jacobian_args = list()
        self.g_jacobian_args = list()
        self.jacob_states = list()
        self.jacob_algebs = list()
        self.jacobian_store_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}
        self.jacobian_store_equations = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}

        # Converts model variables into symbolic expressions.
        self.sym_state_vars = list()
        self.sym_algeb_vars = list()

        """
        Fill in the structures
        """
        self.x0 = np.zeros(len(dynamic_model.stat_var) + len(dynamic_model.output_state_var))
        self.t_const0 = np.zeros(len(dynamic_model.stat_var) + len(dynamic_model.output_state_var))
        self.y0 = np.zeros(len(dynamic_model.algeb_var)) + len(dynamic_model.output_algeb_var)

        index = 0

        for i, (key, elem) in enumerate(dynamic_model.stat_var.items()):
            self.real_state_vars.append(elem)
            self.variables.append(elem)
            self.variables_list.append(elem.name)
            self.vars_index[index] = elem.name

            self.x0[i] = elem.init_val
            self.t_const0[i] = elem.t_const

            index += 1
            self.nx += 1


            if elem.eq is not None:
                self.state_eqs.append(elem)
                self.eqs_list.append(elem.name)
            self.state_vars.append(elem)
            self.vars_list.append(elem.name)
            self.internal_vars.append(elem.name)

        for i, (key, elem) in enumerate(dynamic_model.algeb_var.items()):
            self.real_algeb_vars.append(elem)
            self.variables.append(elem)
            self.variables_list.append(elem.name)
            self.vars_index[index] = elem.name


            self.y0[i] = elem.init_val

            index += 1
            self.ny += 1

            if elem.eq is not None:
                self.algeb_eqs.append(elem)
                self.eqs_list.append(elem.name)
            self.algeb_vars.append(elem)
            self.vars_list.append(elem.name)
            self.internal_vars.append(elem.name)

        for key, elem in dynamic_model.input_state_var.items():
            self.variables.append(elem)
            self.variables_list.append(elem.name)
            self.vars_index[index] = elem.name
            index += 1

            if elem.eq is not None:
                self.input_state_eqs.append(elem)
                self.eqs_list.append(elem.name)
            self.state_vars.append(elem)
            self.vars_list.append(elem.name)
            self.input_vars.append(elem)

        for key, elem in dynamic_model.input_algeb_var.items():
            self.variables.append(elem)
            self.variables_list.append(elem.name)
            self.vars_index[index] = elem.name
            index += 1

            if elem.eq is not None:
                self.input_algeb_eqs.append(elem)
                self.eqs_list.append(elem.name)
            self.algeb_vars.append(elem)
            self.vars_list.append(elem.name)
            self.input_vars.append(elem)

        for key, elem in dynamic_model.output_state_var.items():
            self.real_state_vars.append(elem)
            self.variables.append(elem)
            self.variables_list.append(elem.name)
            self.vars_index[index] = elem.name
            index += 1

            if elem.eq is not None:
                self.state_eqs.append(elem)
                self.eqs_list.append(elem.name)
            self.state_vars.append(elem)
            self.vars_list.append(elem.name)
            self.output_vars.append(elem)

        for key, elem in dynamic_model.output_algeb_var.items():
            self.real_algeb_vars.append(elem)
            self.variables.append(elem)
            self.variables_list.append(elem.name)
            self.vars_index[index] = elem.name
            index += 1

            if elem.eq is not None:
                self.algeb_eqs.append(elem)
                self.eqs_list.append(elem.name)
            self.algeb_vars.append(elem)
            self.vars_list.append(elem.name)
            self.output_vars.append(elem)

        folder_to_save = get_create_dynamics_model_folder(grid_id=grid_id)
        file_path = self.generate(folder_to_save=folder_to_save)  # does all the symbolic operations needed

        self._variables_names_for_ordering = self.variables_names_for_ordering
        self._jacobian_info = self.jacobian_store_info
        self._jacobian_equations = self.jacobian_store_equations
        self._variables_names_for_ordering_f = self._variables_names_for_ordering['f']
        self._variables_names_for_ordering_g = self._variables_names_for_ordering['g']

        self._module = load_file_as_module(file_path=file_path)
        self._f_update_ptr = load_function_from_module(module=self._module, function_name="f_update")
        self._g_update_ptr = load_function_from_module(module=self._module, function_name="g_update")
        self._f_ia_ptr = load_function_from_module(module=self._module, function_name="f_ia")
        self._g_ia_ptr = load_function_from_module(module=self._module, function_name="g_ia")

    def generate(self, folder_to_save):
        """
        Perform full symbolic processing pipeline.

        Steps:
            1. Generate symbolic variables
            2. Convert string equations into symbolic expressions
            3. Compute and store Jacobian matrices
            4. Generate numerical Python code from symbolic forms
        :return:
        """
        self.generate_symbols()
        self.generate_equations()
        self.generate_jacobians()
        return self.generate_code(folder_to_save)

    def generate_symbols(self):
        """
        Converts model variables into symbolic expressions.
        :return:
        """
        # Define symbolic variables
        self.sym_state_vars = [sym.Symbol(v.name) for v in self.state_vars]
        self.sym_algeb_vars = [sym.Symbol(v.name) for v in self.algeb_vars]

    def generate_equations(self):
        """
         Convert string equations into symbolic expressions.

        This method:
            - Parses model equations using SymPy
            - Identifies symbols used in each equation
            - Stores arguments used in f/g functions
            - Creates lambdified numerical functions for f and g
        :return:
        """
        variables_f_g = [self.state_eqs, self.algeb_eqs]
        input_variables_f_g = [self.input_state_eqs, self.input_algeb_eqs]
        equations_f_g = [self.f_list, self.g_list]
        equation_type = ['f', 'g']

        if self.name == "Bus":
            variables_f_g = input_variables_f_g

        for variables, equations, eq_type in zip(variables_f_g, equations_f_g, equation_type):
            # list with the information of the order of the equations in the output of f_update and g_update
            variables_names_for_ordering = []

            # create a list with all symbolic equations (symbolic_expr) and a list with all symbols in equations (symbols_in_equ)
            symbolic_eqs = []
            symbolic_vars = []
            for var in variables:

                if var.eq:
                    variables_names_for_ordering.append(var.name)
                    symbolic_expr = sym.sympify(var.eq)
                    symbols_in_eq = symbolic_expr.free_symbols

                    symbolic_eqs.append(symbolic_expr)
                    equations.append(symbolic_expr)

                    for symb in symbols_in_eq:
                        if symb not in symbolic_vars:
                            symbolic_vars.append(symb)

            symbolic_args = sorted(symbolic_vars, key=lambda s: s.name)

            # store arguments for f and g functions
            for arg in symbolic_args:
                if eq_type == 'f':
                    self.f_args.append(str(arg))
                else:
                    self.g_args.append(str(arg))

            # Lambdify numerical evaluation functions
            self.lambda_equations[eq_type] = lambdify(symbolic_args, sym.Matrix(symbolic_eqs), modules='numpy')
            self.variables_names_for_ordering[eq_type] = variables_names_for_ordering

        self.f_matrix = sym.Matrix(self.f_list)
        self.g_matrix = sym.Matrix(self.g_list)

    def generate_jacobians(self):
        """
        Compute and lambdify Jacobian matrices for f and g equations.
        :return:
        """
        sym_variables = self.sym_state_vars + self.sym_algeb_vars
        all_variables = self.state_vars + self.algeb_vars

        # Compute Jacobian matrices
        f_jacobian_symbolic = self.f_matrix.jacobian(sym_variables) if len(self.f_matrix) > 0 else sym.Matrix([])
        g_jacobian_symbolic = self.g_matrix.jacobian(sym_variables) if len(self.g_matrix) > 0 else sym.Matrix([])

        # Extract unique symbols
        f_jac_symbols = list(f_jacobian_symbolic.free_symbols)
        g_jac_symbols = list(g_jacobian_symbolic.free_symbols)

        # Convert to sparse matrices
        f_jacob_symbolic_spa = sym.SparseMatrix(f_jacobian_symbolic)
        g_jacob_symbolic_spa = sym.SparseMatrix(g_jacobian_symbolic)

        # Store Jacobian information
        for idx, eq_sparse in enumerate([f_jacob_symbolic_spa, g_jacob_symbolic_spa]):
            for e_idx, v_idx, e_symbolic in eq_sparse.row_list():
                var_type = all_variables[v_idx].var_type
                eq_var_code = f"d{['f', 'g'][idx]}{var_type.value}"
                if idx == 0:
                    self.jacobian_store_info[eq_var_code].append((e_idx, v_idx))
                    self.jacobian_store_equations[eq_var_code].append(str(e_symbolic))
                else:
                    self.jacobian_store_info[eq_var_code].append((e_idx + len(self.state_eqs), v_idx))
                    self.jacobian_store_equations[eq_var_code].append(str(e_symbolic))

        # store arguments for f_jacobian
        f_jac_args = sorted(f_jac_symbols, key=lambda s: s.name)
        for arg in f_jac_args:
            self.f_jac_arguments.append(str(arg))

        # store arguments for g_jacobian
        g_jac_args = sorted(g_jac_symbols, key=lambda s: s.name)
        for arg in g_jac_args:
            self.g_jac_arguments.append(str(arg))

        # Lambdify Jacobian functions
        self.jacob_states = lambdify(f_jac_args, tuple(f_jacobian_symbolic), modules='numpy')
        self.jacob_algebs = lambdify(g_jac_args, tuple(g_jacobian_symbolic), modules='numpy')

    def _rename_func(self, func, func_name, vars=False):
        """
        Generate source code from a lambdified function, renaming it for clarity.
        :param func: The lambdified function object
        :param func_name: Desired name for the generated function
        :param vars: Extra arguments to append to the function signature
        :return: Modified function source code string
        """
        if func is None:
            return f"# empty {func_name}\n"

        src = inspect.getsource(func).replace("def _lambdifygenerated(", f"def {func_name}(")
        src = src.replace("Indicator", "")  # Remove indicator
        if vars:
            src = src.replace("):", ", " + ', '.join(vars) + "):")

        return src + '\n'

    def generate_code(self, folder_to_save):
        """
        Write numerical model code to a Python file.

        This method:
            - Evaluation functions for f, g
            - Jacobian evaluation functions
            - Argument names
            - Variable ordering metadata
            - Sparsity pattern (Jacobian info)
        :return: The path to the generated Python file
        """
        filename = f"{self.idtag}.py"
        file_path = os.path.join(folder_to_save, filename)

        with open(file_path, 'w') as f:

            f.write("from numba import njit\n")
            f.write("from numpy import *\n\n")

            f.write("'''\n")
            f.write(f"Generated by GridCal {datetime.datetime.now()}\n")
            f.write(f"{self.name}\n")
            f.write("'''\n\n")

            for eq_type, func_name in [('f', 'f_update'), ('g', 'g_update')]:
                py_expr = self._rename_func(self.lambda_equations.get(eq_type), func_name)
                f.write(f"@njit(cache=True)\n")
                f.write(f"{py_expr}")

            f.write(f"f_args =" + pprint.pformat(sorted(self.f_args), width=1000) + '\n')
            f.write(f"g_args =" + pprint.pformat(sorted(self.g_args), width=1000) + '\n\n')

            f.write(f"variables_names_for_ordering =" + pprint.pformat(self.variables_names_for_ordering,
                                                                       width=1000) + '\n\n')

            for name, func in [('f', self.jacob_states), ('g', self.jacob_algebs)]:
                py_expr = self._rename_func(func, f"{name}_ia")
                f.write(f"@njit(cache=True)\n")
                f.write(f"{py_expr}")

            f.write(f"f_jac_args =" + pprint.pformat(self.f_jac_arguments, width=1000) + '\n')
            f.write(f"g_jac_args =" + pprint.pformat(self.g_jac_arguments, width=1000) + '\n\n')

            f.write(f"jacobian_info = {self.jacobian_store_info}" + '\n')
            f.write(f"jacobian_equations =" + pprint.pformat(self.jacobian_store_equations, width=1000))

        return file_path

    def calc_f_g_functions(self):
        """
        calculates f and g functions and gets variables order.
        :return: f and g functions values, lists with the order of the variables for f and g
        """
        len_state_eq = len(self.state_eqs)
        len_algeb_eq = len(self.algeb_eqs)
        if self.name == "Bus":
            len_state_eq = len(self.input_state_eqs)
            len_algeb_eq = len(self.input_algeb_eqs)


        f_values_device = np.zeros((self.n, len_state_eq))
        g_values_device = np.zeros((self.n, len_algeb_eq))

        for i in range(self.n):
            # get f values
            if self.f_args:
                f_values = self._f_update_ptr(*self.f_input_values[i])
                for j in range(len_state_eq):
                    f_values_device[i][j] = f_values[j]

            # get g values
            if self.g_args:
                g_values = self._g_update_ptr(*self.g_input_values[i])
                for j in range(len_algeb_eq):
                    g_values_device[i][j] = g_values[j]

        return (f_values_device, g_values_device,
                self._variables_names_for_ordering_f, self._variables_names_for_ordering_g)

    def calc_local_jacs(self):
        """

        :return:
        """
        f_jacobians = np.zeros((self.n, len(self.state_eqs), len(self.variables_list)))
        g_jacobians = np.zeros((self.n, len(self.algeb_eqs), len(self.variables_list)))
        state_eqs = self.state_eqs
        algeb_eqs = self.algeb_eqs

        if self.name == "Bus":
            f_jacobians = np.zeros((self.n, len(self.input_state_eqs), len(self.variables_list)))
            g_jacobians = np.zeros((self.n, len(self.input_algeb_eqs), len(self.variables_list)))
            state_eqs = self.input_state_eqs
            algeb_eqs = self.input_algeb_eqs

        for i in range(self.n):

            if self.f_jac_arguments:
                local_jac_f = self._f_ia_ptr(*self.f_jac_input_values[i])
                for j, funct in enumerate(state_eqs):
                    for k, var in enumerate(self.variables_list):
                        f_jacobians[i, j, k] = local_jac_f[j * len(self.variables_list) + k]

            if self.g_jac_arguments:
                local_jac_g = self._g_ia_ptr(*self.g_jac_input_values[i])
                for j, funct in enumerate(algeb_eqs):
                    for k, var in enumerate(self.variables_list):
                        g_jacobians[i, j, k] = local_jac_g[j * len(self.variables_list) + k]

        jacobian = np.concatenate((f_jacobians, g_jacobians), axis=1)

        return jacobian, self._jacobian_info, self._jacobian_equations
