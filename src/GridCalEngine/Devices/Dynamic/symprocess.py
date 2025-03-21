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
    def __init__(self, model):
        self.device = model
        self.spoint = model.spoint

        self.sym_num_params =[]
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
        self.f_matrix = []
        self.g_matrix = []
        self.f_jacob_sym = sp.Matrix([])
        self.g_jacob_sym = sp.Matrix([])
        self.symb_vars_dict = {}
        self.lambda_equations = {}


    def generate(self):
        self.generate_symbols()
        self.generate_equations()
        self.generate_pycode()

    def _rename_func(self, func, func_name, vars):
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
        right_parenthesis = ', '.join(vars) + "):"
        src = src.replace("):", right_parenthesis)

        src += '\n'
        return src

    def generate_symbols(self):
        # Convert strings to symbolic expressions

        # Define symbolic parameters
        self.sym_num_params = [sp.Symbol(param.symbol) for param in self.spoint.numdynParam]
        self.sym_idx_params = [sp.Symbol(param.symbol) for param in self.spoint.idxdynParam]
        self.sym_ext_params = [sp.Symbol(param.symbol) for param in self.spoint.extdynParam]

        # Define symbolic variables
        self.sym_state = [sp.Symbol(v.symbol) for v in self.spoint.statVars]
        self.sym_algeb = [sp.Symbol(v.symbol) for v in self.spoint.algebVars]
        self.sym_extern = [sp.Symbol(v.symbol) for v in self.spoint.externAlgebs]
        self.sym_aliasalgeb = [sp.Symbol(v.symbol) for v in self.spoint.aliasAlgebs]
        self.sym_externstate = [sp.Symbol(v.symbol) for v in self.spoint.externStates]
        self.sym_aliasstate = [sp.Symbol(v.symbol) for v in self.spoint.aliasStats]
        self.sym_externvars = [sp.Symbol(v.symbol) for v in self.spoint.externVars]

    def generate_equations(self):
        """Parses multiple equations, computes Jacobians, and generates Python files."""

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
            self.lambda_equations[eq_type] = lambdify(var_symb, tuple(eq_symb), modules='numpy')
        self.f_matrix = sp.Matrix(self.f_list)
        self.g_matrix = sp.Matrix(self.g_list)
        print(self.g_matrix)

    #def generate_jacobians(self):
     #   # first we call the g and f matrix, where the symbolic equations are stored in, get the jacobians with sp.jacobian, convert the resulting jacobian matrices to sparce matrices and build a list with both matrices for f and g.
      #  sym_variables = [self.sym_state]+[self.sym_algeb]

       # self.g_jacob_sym = self.g_matrix.jacobian(sym_variables)



        return

    def generate_pycode(self):

        pycode_path = get_pycode_path()
        filename = f"{self.spoint.name}.py"
        file_path = os.path.join(pycode_path, filename)
        with open(file_path, 'w') as f:

            # write imports
            f.write("import numpy\n\n")

            # write f_equations
            py_expr = self._rename_func(self.lambda_equations['f'], 'f_update', self.spoint.stats_symb)
            f.write(f"{py_expr}\n")

            # write g_equations
            py_expr = self._rename_func(self.lambda_equations['g'], 'g_update', self.spoint.algebs_symb)
            f.write(f"{py_expr}\n")
        return file_path

    def generate_jacobian(self):

        #
        return









        #f_expressions = [sp.sympify(expr) for expr in self.spoint.f]
        #if len(self.spoint.g) != 0:
            #g_expressions = [sp.sympify(expr) for expr in self.spoint.g if expr != '']

        # Compute Jacobians
        #jacobian_f = [[sp.diff(expr, var) for var in stateSymb] for expr in f_expressions]
        #jacobian_g = [[sp.diff(expr, var) for var in algebSymb] for expr in g_expressions]

        ## -------- Python Code Generation -------- #
        #pycode_path = get_pycode_path()
        #filename = f"{self.spoint.name}.py"
        #file_path = os.path.join(pycode_path, filename)
        #with open(file_path, 'w') as f:
            ## Write imports
            #f.write("import numpy\n\n")

            ## Write f_expressions
            #f.write("def f_update(*args):\n")
            #for i, var in enumerate(stateSymb):
                #f.write(f"    {var} = args[{i}]\n")
            #for idx, expr in enumerate(f_expressions):
                #py_expr = self._rename_func(lambdify(stateSymb, expr, modules='numpy'),'f_update')
               # f.write(f"    expr{idx} = {py_expr}\n")
            #f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(f_expressions))])}]\n\n")

            # g_update function
            #lambdifyed = lambdify(algebSymb, tuple(g_expressions), modules='numpy')

            #py_expr = self._rename_func(lambdifyed, 'g_update', self.spoint.algebVars)
            #f.write(f"{py_expr}\n")


            # # f Jacobians
            #f.write("def jac_f_update(*args):\n")
            #for i, var in enumerate(stateSymb):
                #f.write(f"    {var} = args[{i}]\n")
            #for idx, expr in enumerate(jacobian_f):
                #py_expr = lambdify(stateSymb, expr, modules="numpy")
                #f.write(f"    expr{idx} = {py_expr}\n")
            #f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(jacobian_f))])}]\n\n")

            ## g Jacobians
            #f.write("def jac_g_update(*args):\n")
            #for i, var in enumerate(algebSymb):
               # f.write(f"    {var} = args[{i}]\n")
            #for idx, expr in enumerate(jacobian_g):
               # py_expr = lambdify(algebSymb, expr, modules="numpy")
                #f.write(f"    expr{idx} = {py_expr}\n")
            #f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(jacobian_g))])}]\n\n")

        #return file_path
