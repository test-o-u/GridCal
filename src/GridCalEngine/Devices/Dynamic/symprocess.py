# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import numpy
import os
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy import Tuple
import numpy as np
import inspect

from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path

select_args_add = ["__zeros", "__ones", "__falses", "__trues"]


class Symprocess:
    def __init__(self, model):
        self.model = model
        self.spoint = model.spoint

        self.sym_num_params = []
        self.sym_idx_params = []
        self.sym_ext_params = []

        self.sym_state = []
        self.sym_algeb = []
        self.sym_extern = []
        self.sym_aliasalgeb = []
        self.sym_externstate = []
        self.sym_aliasstate = []
        self.sym_externvars = []

        self.f_list = []
        self.g_list = []
        self.f_matrix = ()
        self.g_matrix = ()
        self.f_jacob_sym = sp.Matrix([])
        self.g_jacob_sym = sp.Matrix([])
        self.symb_vars_dict = {}
        self.lambda_equations = {}

        self.jacob_states = []
        self.jacob_algebs = []
        self.all_variables = []
        self.sym_variables = []
        self.f_jac_symbols = []
        self.g_jac_symbols = []

        self.jacobian_store_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}

    def generate(self):
        self.generate_symbols()
        self.generate_equations()
        self.generate_jacobians()
        self.generate_pycode()

    def _rename_func(self, func, func_name, vars=False):
        """
        Rename the function name and return source code.

        This function performs these tasks:

        1. rename ``_lambdifygenerated`` to the given ``func_name``.
        2. append four arguments if ``select`` is used to pass numba
           compilation.
        3. remove ``Indicator`` for wrappers of logic expressions.

        This function does not check for name conflicts. Install `yapf` for
        optional code reformatting (takes extra processing time).

        It also patches function argument list for select.
        """
        if func is None:
            return f"# empty {func_name}\n"

        src = inspect.getsource(func)
        src = src.replace("def _lambdifygenerated(", f"def {func_name}(")

        # remove `Indicator`
        src = src.replace("Indicator", "")

        # append additional arguments
        if vars:
            right_parenthesis = ', '.join(vars) + "):"
            src = src.replace("):", right_parenthesis)

        src += '\n'
        return src

    def generate_symbols(self):
        " Convert strings to symbolic expressions"

        # Define symbolic parameters
        self.sym_num_params = [sp.Symbol(param.symbol) for param in self.spoint.numdynParam]
        self.sym_idx_params = [sp.Symbol(param.symbol) for param in self.spoint.idxdynParam]
        self.sym_ext_params = [sp.Symbol(param.symbol) for param in self.spoint.extdynParam]

        # Define symbolic variables
        self.sym_state = [sp.Symbol(v.symbol) for v in self.spoint.stats]
        self.sym_algeb = [sp.Symbol(v.symbol) for v in self.spoint.algebs]
        self.sym_extern = [sp.Symbol(v.symbol) for v in self.spoint.externAlgebs]
        self.sym_aliasalgeb = [sp.Symbol(v.symbol) for v in self.spoint.aliasAlgebs]
        self.sym_externstate = [sp.Symbol(v.symbol) for v in self.spoint.externStates]
        self.sym_aliasstate = [sp.Symbol(v.symbol) for v in self.spoint.aliasStats]
        self.sym_externvars = [sp.Symbol(v.symbol) for v in self.spoint.externVars]

    def generate_equations(self):
        """compute lambdifyed equations"""

        variables = [self.spoint.stats, self.spoint.algebs]
        equations_f_g = [self.f_list, self.g_list]
        equation_type = ['f', 'g']
        expr_list = [self.f_list, self.g_list]

        for var_list, equations, eq_type in zip(variables, equations_f_g, equation_type):
            eq_symb = []
            var_symb = []
            for var in var_list:
                if var.eq != '':
                    symb_expr = sp.sympify(var.eq)

                    symb_var = symb_expr.free_symbols

                    eq_symb.append(symb_expr)
                    equations.append(symb_expr)
                    for symb in symb_var:
                        if symb not in var_symb:
                            var_symb.append(symb)
                    if eq_type == 'f':
                        self.f_list.append(symb_expr)
                    else:
                        self.g_list.append(symb_expr)
            self.lambda_equations[eq_type] = lambdify(var_symb, sp.Matrix(eq_symb), modules = 'numpy')
        self.f_matrix = sp.Matrix(self.f_list)
        self.g_matrix = sp.Matrix(self.g_list)

    def generate_jacobians(self):

        jacob_states = []
        jacob_algebs = []
        f_jacob_sym = sp.Matrix([])
        g_jacob_sym = sp.Matrix([])

        # call the g and f matrix, where the symbolic equations are stored in, get the jacobians with sp.jacobian, convert the resulting jacobian matrices to sparce matrices and build a list with both matrices for f and g.
        self.sym_variables = self.sym_state + self.sym_algeb
        self.all_variables = self.spoint.stats + self.spoint.algebs

        if len(self.f_matrix) > 0:
            f_jacob_sym = self.f_matrix.jacobian(self.sym_variables)
        f_jacobian_free_symbols = f_jacob_sym.free_symbols
        for sym in f_jacobian_free_symbols:
            if sym not in self.f_jac_symbols:
                self.f_jac_symbols.append(sym)

        if len(self.g_matrix) > 0:
            g_jacob_sym = self.g_matrix.jacobian(self.sym_variables)
        g_jacobian_free_symbols = g_jacob_sym.free_symbols
        for sym in g_jacobian_free_symbols:
            if sym not in self.g_jac_symbols:
                self.g_jac_symbols.append(sym)

        f_jacob_sym_spa = sp.SparseMatrix(f_jacob_sym)
        g_jacob_sym_spa = sp.SparseMatrix(g_jacob_sym)

        fg_jacob_sym_spa = [f_jacob_sym_spa, g_jacob_sym_spa]

        # generate sparse matrix, save non zero and lambdify

        for idx, eq_sparse in enumerate(fg_jacob_sym_spa):
            for item in eq_sparse.row_list():
                e_idx, v_idx, e_symbolic = item
                if idx == 0:
                    jacob_states.append(e_symbolic)
                    var_type = self.all_variables[v_idx].var_type

                    eq_var_code = 'df' + str(var_type)

                    self.jacobian_store_info[eq_var_code].append((e_idx, v_idx))

                else:
                    jacob_algebs.append(e_symbolic)

                    var_type = self.all_variables[v_idx].var_type

                    eq_var_code = 'dg' + str(var_type)

                    self.jacobian_store_info[eq_var_code].append((e_idx, v_idx))
        self.jacob_states = sp.lambdify(self.f_jac_symbols, sp.Matrix(jacob_states), modules= 'numpy')
        self.jacob_algebs = sp.lambdify(self.g_jac_symbols, sp.Matrix(jacob_algebs), modules= 'numpy')

        return

    def generate_pycode(self):
        pycode_path = get_pycode_path()
        filename = f"{self.spoint.name}.py"
        file_path = os.path.join(pycode_path, filename)
        with open(file_path, 'w') as f:
            # write imports
            f.write("import numpy\n\n")
            f.write("from numpy import *\n\n")
            # write f_equations
            py_expr = self._rename_func(self.lambda_equations['f'], 'f_update')
            f.write(f"{py_expr}\n")

            # write g_equations
            py_expr = self._rename_func(self.lambda_equations['g'], 'g_update')
            f.write(f"{py_expr}\n")

            # jacobians
            name = 'f'
            py_expr = self._rename_func(self.jacob_states, f'{name}_ia')
            f.write(f"{py_expr}\n")
            name = 'g'
            py_expr = self._rename_func(self.jacob_algebs, f'{name}_ia')
            f.write(f"{py_expr}\n")

            f.write(f"jacobian_info = {self.jacobian_store_info}\n")
        return file_path
