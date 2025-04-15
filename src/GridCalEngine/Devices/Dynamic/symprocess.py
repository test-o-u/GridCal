# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import sympy as sp
import inspect
import pprint
from sympy.utilities.lambdify import lambdify
from GridCalEngine.Devices.Dynamic.utils.paths import get_generated_module_path


class SymProcess:
    """
    Handles symbolic processing for dynamic models.

    This class:
    - Converts model equations into symbolic form.
    - Generates symbolic-to-numeric transformation.
    - Computes and stores Jacobian matrices.
    - Generates Python code for compiled symbolic models.
    """

    def __init__(self, model):
        """
        Initialize symbolic processing.

        Args:
            model: The model instance containing symbolic equations.
        """
        self.model = model

        # Symbolic Parameters
        self.sym_num_params = []
        self.sym_idx_params = []

        # Symbolic Variables
        self.sym_state = []
        self.sym_algeb = []

        # Symbolic Equations
        self.f_args = []
        self.g_args = []
        self.f_list = []
        self.g_list = []
        self.f_matrix = ()
        self.g_matrix = ()
        self.lambda_equations = {}
        self.variables_names_for_ordering = {}

        # Jacobians
        self.f_jacobian_args = []
        self.g_jacobian_args = []
        self.jacob_states = []
        self.jacob_algebs = []
        self.jacobian_store_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}

    def generate(self):
        """
        Generates symbolic representations, equations, Jacobians, and Python code.
        """
        self.generate_symbols()
        self.generate_equations()
        self.generate_jacobians()
        self.generate_code()

    def generate_symbols(self):
        """
        Converts model parameters and variables into symbolic expressions.
        """
        # Define symbolic variables
        self.sym_state = [sp.Symbol(v.symbol) for v in self.model.stats]
        self.sym_algeb = [sp.Symbol(v.symbol) for v in self.model.algebs]

    def generate_equations(self):
        """
        Converts string equations into symbolic expressions and lambdifies them.
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
        Computes symbolic Jacobian matrices and lambdifies them.
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
        count = 0
        for idx, eq_sparse in enumerate([f_jacob_symbolic_spa, g_jacob_symbolic_spa]):
            for e_idx, v_idx, e_symbolic in eq_sparse.row_list():
                var_type = all_variables[v_idx].var_type
                eq_var_code = f"d{['f', 'g'][idx]}{var_type}"
                if idx == 0:
                    self.jacobian_store_info[eq_var_code].append((e_idx, v_idx))

                    if var_type == 'x':
                        count += 1
                else:
                    self.jacobian_store_info[eq_var_code].append((e_idx + count, v_idx))

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
        Renames a lambdified function for improved clarity.
        Args:
            func: The function to rename.
            func_name (str): The desired function name.
            vars (list, optional): Additional arguments to append.
        Returns:
            str: The modified function source code.
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
        Generates Python code for numerical model evaluation.
        """
        generated_module_path = get_generated_module_path()
        filename = f"{self.model.name}.py"
        file_path = os.path.join(generated_module_path, filename)

        with open(file_path, 'w') as f:
            f.write("import numpy\n\nfrom numpy import *\n\n")

            for eq_type, func_name in [('f', 'f_update'), ('g', 'g_update')]:
                py_expr = self._rename_func(self.lambda_equations.get(eq_type), func_name)
                f.write(f"{py_expr}\n")

            f.write(f"f_args =" + pprint.pformat(sorted(self.f_args)) + '\n')
            f.write(f"g_args =" + pprint.pformat(sorted(self.g_args)) + '\n')

            f.write(f"variables_names_for_ordering =" + pprint.pformat(self.variables_names_for_ordering) + '\n')

            for name, func in [('f', self.jacob_states), ('g', self.jacob_algebs)]:
                py_expr = self._rename_func(func, f"{name}_ia")
                f.write(f"{py_expr}\n")

            f.write(f"f_jac_args =" + pprint.pformat(self.f_jacobian_args) + '\n')
            f.write(f"g_jac_args =" + pprint.pformat(self.g_jacobian_args) + '\n')

            f.write(f"jacobian_info = {self.jacobian_store_info}")

        return file_path
