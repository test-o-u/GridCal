# TODO: integrate:)
import numpy
import os
import sympy as sp
from sympy.utilities.lambdify import lambdify
import numpy as np
import inspect

from GridCalEngine.Devices.Dynamic.utils.paths import get_pycode_path

select_args_add = ["__zeros", "__ones", "__falses", "__trues"]

class Symprocess:
    def __init__(self, device):
        self.device = device
        self.spoint = device.spoint

    def _rename_func(self, func, func_name, yapf_pycode=False):
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

        # append additional arguments for select
        if 'select' in src:
            right_parenthesis = ", " + ', '.join(select_args_add) + "):"
            src = src.replace("):", right_parenthesis)

        #if yapf_pycode:
         #   try:
          #      from yapf.yapflib.yapf_api import FormatCode
           #     src = FormatCode(src, style_config='pep8')[0]  # drop the encoding `None`
           # except ImportError:
            #    logger.warning("`yapf` not installed. Skipped code reformatting.")

        src += '\n'
        return src
    def generate(self):
        """Parses multiple equations, computes Jacobians, and generates Python files."""
        # Convert strings to symbolic expressions
        print(self.device)
        print(self.spoint.f)
        print(self.spoint.g)


        # Define symbolic parameters
        num_params = [sp.Symbol(param) for param in self.spoint.numdynParam]
        idx_params = [sp.Symbol(param) for param in self.spoint.idxdynParam]
        ext_params = [sp.Symbol(param) for param in self.spoint.extdynParam]

        # Define symbolic variables


        state_vars = [sp.Symbol(v) for v in self.spoint.statVars]
        algeb_vars = [sp.Symbol(v) for v in self.spoint.algebVars]

        f_expressions = [sp.sympify(expr) for expr in self.spoint.f]
        if len(self.spoint.g) != 0:
            g_expressions = [sp.sympify(expr) for expr in self.spoint.g if expr != '']

        # Compute Jacobians
        jacobian_f = [[sp.diff(expr, var) for var in state_vars] for expr in f_expressions]
        jacobian_g = [[sp.diff(expr, var) for var in algeb_vars] for expr in g_expressions]

        # -------- Python Code Generation -------- #
        pycode_path = get_pycode_path()
        filename = f"{self.spoint.name}.py"
        file_path = os.path.join(pycode_path, filename)
        with open(file_path, 'w') as f:
            # Write imports
            f.write("import numpy\n\n")

            # Write f_expressions
            f.write("def f_update(*args):\n")
            for i, var in enumerate(state_vars):
                f.write(f"    {var} = args[{i}]\n")
            for idx, expr in enumerate(f_expressions):
                py_expr = self._rename_func(lambdify(state_vars, expr, modules='numpy'),'f_update')
                f.write(f"    expr{idx} = {py_expr}\n")
            f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(f_expressions))])}]\n\n")

            # g_update function
            f.write("def g_update(*args):\n")
            for i, var in enumerate(algeb_vars):
                f.write(f"    {var} = args[{i}]\n")
            for idx, expr in enumerate(g_expressions):
                py_expr = self._rename_func(lambdify(algeb_vars, expr, modules='numpy'), 'g_update')
                f.write(f"    expr{idx} = {py_expr}\n")
            f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(g_expressions))])}]\n\n")

            # # f Jacobians
            f.write("def jac_f_update(*args):\n")
            for i, var in enumerate(state_vars):
                f.write(f"    {var} = args[{i}]\n")
            for idx, expr in enumerate(jacobian_f):
                py_expr = lambdify(state_vars, expr, modules="numpy")
                f.write(f"    expr{idx} = {py_expr}\n")
            f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(jacobian_f))])}]\n\n")

            # g Jacobians
            f.write("def jac_g_update(*args):\n")
            for i, var in enumerate(algeb_vars):
                f.write(f"    {var} = args[{i}]\n")
            for idx, expr in enumerate(jacobian_g):
                py_expr = lambdify(algeb_vars, expr, modules="numpy")
                f.write(f"    expr{idx} = {py_expr}\n")
            f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(jacobian_g))])}]\n\n")

        return file_path
