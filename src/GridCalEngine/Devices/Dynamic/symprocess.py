# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import inspect
import pdb
import pprint
import sympy as sp
from sympy.utilities.lambdify import lambdify
from GridCalEngine.Devices.Dynamic.utils.paths import get_generated_module_path


class SymProcess:
    """
    Handles symbolic processing for dynamic models.

    Responsibilities:
        - Converts model equations into symbolic form
        - Generates lambdified numerical functions
        - Computes Jacobian matrices symbolically
        - Exports Python code for numerical model evaluation
    """

    def __init__(self, model):
        """
        SymProcess class constructor
        :param model: The model instance containing variables and equations
        """
        self.model = model

        # Symbolic Parameters
        self.sym_num_params = list()
        self.sym_idx_params = list()

        # Symbolic Variables
        self.sym_state = list()
        self.sym_algeb = list()

        # Symbolic Equations
        self.f_args = list()
        self.g_args = list()
        self.f_list = list()
        self.g_list = list()
        self.f_matrix = ()
        self.g_matrix = ()
        self.lambda_equations = {}
        self.variables_names_for_ordering = {}

        # Jacobians
        self.f_jacobian_args = list()
        self.g_jacobian_args = list()
        self.jacob_states = list()
        self.jacob_algebs = list()
        self.jacobian_store_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}
        self.jacobian_store_equations = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}

    def generate(self):
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
        self.generate_code()

    def generate_symbols(self):
        """
        Converts model variables into symbolic expressions.
        :return:
        """
        # Define symbolic variables
        self.sym_state = [sp.Symbol(v.symbol) for v in self.model.stats]
        self.sym_algeb = [sp.Symbol(v.symbol) for v in self.model.algebs]

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
        variables_f_g = [self.model.stats, self.model.algebs]
        equations_f_g = [self.f_list, self.g_list]
        equation_type = ['f', 'g']

        for variables, equations, eq_type in zip(variables_f_g, equations_f_g, equation_type):
            # list with the information of the order of the equations in the output of f_update and g_update
            variables_names_for_ordering = []

            # create a list with all symbolic equations (symbolic_expr) and a list with all symbols in equations (symbols_in_equ)
            symbolic_eqs = []
            symbolic_vars = []
            for var in variables:
                if var.eq:
                    variables_names_for_ordering.append(var.name)
                    symbolic_expr = sp.sympify(var.eq)
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
            self.lambda_equations[eq_type] = lambdify(symbolic_args, sp.Matrix(symbolic_eqs), modules='numpy')
            self.variables_names_for_ordering[eq_type] = variables_names_for_ordering

        self.f_matrix = sp.Matrix(self.f_list)
        self.g_matrix = sp.Matrix(self.g_list)

    def generate_jacobians(self):
        """
        Compute and lambdify Jacobian matrices for f and g equations.
        :return:
        """
        sym_variables = self.sym_state + self.sym_algeb
        all_variables = self.model.stats + self.model.algebs

        # Compute Jacobian matrices
        f_jacobian_symbolic = self.f_matrix.jacobian(sym_variables) if len(self.f_matrix) > 0 else sp.Matrix([])
        g_jacobian_symbolic = self.g_matrix.jacobian(sym_variables) if len(self.g_matrix) > 0 else sp.Matrix([])

        # Extract unique symbols
        f_jac_symbols = list(f_jacobian_symbolic.free_symbols)
        g_jac_symbols = list(g_jacobian_symbolic.free_symbols)

        # Convert to sparse matrices
        f_jacob_symbolic_spa = sp.SparseMatrix(f_jacobian_symbolic)
        g_jacob_symbolic_spa = sp.SparseMatrix(g_jacobian_symbolic)

        # Store Jacobian information
        for idx, eq_sparse in enumerate([f_jacob_symbolic_spa, g_jacob_symbolic_spa]):
            for e_idx, v_idx, e_symbolic in eq_sparse.row_list():
                var_type = all_variables[v_idx].var_type
                eq_var_code = f"d{['f', 'g'][idx]}{var_type}"
                if idx == 0:
                    self.jacobian_store_info[eq_var_code].append((e_idx, v_idx))
                    self.jacobian_store_equations[eq_var_code].append(str(e_symbolic))
                else:
                    self.jacobian_store_info[eq_var_code].append((e_idx + len(self.model.state_vars_list), v_idx))
                    self.jacobian_store_equations[eq_var_code].append(str(e_symbolic))

        # store arguments for f_jacobian
        f_jac_args = sorted(f_jac_symbols, key=lambda s: s.name)
        for arg in f_jac_args:
            self.f_jacobian_args.append(str(arg))

        # store arguments for g_jacobian
        g_jac_args = sorted(g_jac_symbols, key=lambda s: s.name)
        for arg in g_jac_args:
            self.g_jacobian_args.append(str(arg))

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

    def generate_code(self):
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

        generated_module_path = get_generated_module_path()
        filename = f"{self.model.name}.py"
        file_path = os.path.join(generated_module_path, filename)

        with open(file_path, 'w') as f:
            f.write("from numba import njit\n")
            f.write("from numpy import *\n\n")

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

            f.write(f"f_jac_args =" + pprint.pformat(self.f_jacobian_args, width=1000) + '\n')
            f.write(f"g_jac_args =" + pprint.pformat(self.g_jacobian_args, width=1000) + '\n\n')

            f.write(f"jacobian_info = {self.jacobian_store_info}" + '\n')
            f.write(f"jacobian_equations =" + pprint.pformat(self.jacobian_store_equations, width=1000))

        return file_path
