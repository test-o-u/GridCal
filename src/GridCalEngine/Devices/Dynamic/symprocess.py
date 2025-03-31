# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import sympy as sp
import inspect
import pdb
from sympy.utilities.lambdify import lambdify
from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path

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
        self.model_storage = model.model_storage

        # Symbolic Parameters
        self.sym_num_params = []
        self.sym_idx_params = []
        # self.sym_ext_params = []

        # Symbolic Variables
        self.sym_state = []
        self.sym_algeb = []
        # self.sym_extern = []
        # self.sym_aliasalgeb = []
        # self.sym_externstate = []
        # self.sym_aliasstate = []
        # self.sym_externvars = []

        # Symbolic Equations
        self.f_list = []
        self.g_list = []
        self.f_matrix = ()
        self.g_matrix = ()
        self.lambda_equations = {}

        # Jacobians
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
        self.generate_pycode()

    def generate_symbols(self):
        """
        Converts model parameters and variables into symbolic expressions.
        """
        # Define symbolic parameters
        self.sym_num_params = [sp.Symbol(param.symbol) for param in self.model_storage.numdynParam]
        self.sym_idx_params = [sp.Symbol(param.symbol) for param in self.model_storage.idxdynParam]
        # self.sym_ext_params = [sp.Symbol(param.symbol) for param in self.model_storage.extdynParam]

        # Define symbolic variables
        self.sym_state = [sp.Symbol(v.symbol) for v in self.model_storage.stats]
        self.sym_algeb = [sp.Symbol(v.symbol) for v in self.model_storage.algebs]
        # self.sym_aliasalgeb = [sp.Symbol(v.symbol) for v in self.model_storage.aliasAlgebs]
        # self.sym_externstate = [sp.Symbol(v.symbol) for v in self.model_storage.externStates]
        # self.sym_aliasstate = [sp.Symbol(v.symbol) for v in self.model_storage.aliasStats]
        # self.sym_externvars = [sp.Symbol(v.symbol) for v in self.model_storage.externVars]

    def generate_equations(self):
        """
        Converts string equations into symbolic expressions and lambdifies them.
        """
        variables_f_g = [self.model_storage.stats, self.model_storage.algebs]
        equations_f_g = [self.f_list, self.g_list]
        equation_type = ['f', 'g']

        for variables, equations, eq_type in zip(variables_f_g, equations_f_g, equation_type):
            symbolic_eqs = []
            symbolic_vars = []
            #create a list with all symbolic equations (symb_expr) and a list with all symbols in equations (var_symb)
            for var in variables:
                if var.eq:
                    symbolic_expr = sp.sympify(var.eq)
                    symbols_in_eq = symbolic_expr.free_symbols

                    symbolic_eqs.append(symbolic_expr)
                    equations.append(symbolic_expr)

                    for symb in symbols_in_eq:
                        if symb not in symbolic_vars:
                            symbolic_vars.append(symb)

            # Lambdify numerical evaluation functions
            self.lambda_equations[eq_type] = lambdify(symbolic_vars, sp.Matrix(symbolic_eqs), modules='numpy')

        self.f_matrix = sp.Matrix(self.f_list)
        self.g_matrix = sp.Matrix(self.g_list)

    def generate_jacobians(self):
        """
        Computes symbolic Jacobian matrices and lambdifies them.
        """
        sym_variables = self.sym_state + self.sym_algeb
        all_variables = self.model_storage.stats + self.model_storage.algebs

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
                self.jacobian_store_info[eq_var_code].append((e_idx, v_idx))

        # Lambdify Jacobian functions
        self.jacob_states = lambdify(f_jac_symbols, tuple(f_jacobian_symbolic), modules='numpy')
        self.jacob_algebs = lambdify(g_jac_symbols, tuple(g_jacobian_symbolic), modules='numpy')

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

    def generate_pycode(self):
        """
        Generates Python code for numerical model evaluation.
        """
        pycode_path = get_pycode_path()
        filename = f"{self.model_storage.name}.py"
        file_path = os.path.join(pycode_path, filename)

        with open(file_path, 'w') as f:
            f.write("import numpy\n\nfrom numpy import *\n\n")

            for eq_type, func_name in [('f', 'f_update'), ('g', 'g_update')]:
                py_expr = self._rename_func(self.lambda_equations.get(eq_type), func_name)
                f.write(f"{py_expr}\n")

            for name, func in [('f', self.jacob_states), ('g', self.jacob_algebs)]:
                py_expr = self._rename_func(func, f"{name}_ia")
                f.write(f"{py_expr}\n")

            f.write(f"jacobian_info = {self.jacobian_store_info}\n")

        return file_path