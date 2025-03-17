import sympy as sp
import numpy as np
from sympy.printing.pycode import pycode 
from sympy.printing.numpy import NumPyPrinter

def generate(name, model):
    """Parses multiple equations, computes Jacobians, and generates Python files."""
    # Convert strings to symbolic expressions
    f_expressions = [sp.sympify(expr) for expr in model.f]
    g_expressions = [sp.sympify(expr) for expr in model.g]
    
    # Define symbolic variables
    state_vars = [sp.Symbol(v) for v in model.state]
    algeb_vars = [sp.Symbol(v) for v in model.algeb]
    
    # Compute Jacobians
    jacobian_f = [[sp.diff(expr, var) for var in state_vars] for expr in f_expressions]
    jacobian_g = [[sp.diff(expr, var) for var in algeb_vars] for expr in g_expressions]
    
    # -------- Python Code Generation -------- #
    printer = NumPyPrinter()
    filename = f"{name}.py"
    with open(filename, 'w') as f:
        # Write imports
        f.write("import numpy\n\n")

        # Write f_expressions
        f.write("def f_update(*args):\n")
        for i, var in enumerate(state_vars):
            f.write(f"    {var} = args[{i}]\n")
        for idx, expr in enumerate(f_expressions):
            py_expr = printer.doprint(expr)
            f.write(f"    expr{idx} = {py_expr}\n")
        f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(f_expressions))])}]\n\n")

        # g_update function
        f.write("def g_update(*args):\n")
        for i, var in enumerate(algeb_vars):
            f.write(f"    {var} = args[{i}]\n")
        for idx, expr in enumerate(g_expressions):
            py_expr = printer.doprint(expr)
            f.write(f"    expr{idx} = {py_expr}\n")
        f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(g_expressions))])}]\n\n")
        
        # f Jacobians
        f.write("def jac_f_update(*args):\n")
        for i, var in enumerate(state_vars):
            f.write(f"    {var} = args[{i}]\n")
        for idx, expr in enumerate(jacobian_f):
            py_expr = printer.doprint(expr)
            f.write(f"    expr{idx} = {py_expr}\n")
        f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(jacobian_f))])}]\n\n")

        # g Jacobians
        f.write("def jac_g_update(*args):\n")
        for i, var in enumerate(algeb_vars):
            f.write(f"    {var} = args[{i}]\n")
        for idx, expr in enumerate(jacobian_g):
            py_expr = printer.doprint(expr)
            f.write(f"    expr{idx} = {py_expr}\n")
        f.write(f"    return [{', '.join([f'expr{i}' for i in range(len(jacobian_g))])}]\n\n")
        
    return filename 

def generate_pyc(filename):
    """Compiles .py to .pyc."""
    import py_compile
    py_compile.compile(filename, cfile=filename + "c") # cfile: is the location where the bytecode is saved. If not specified, it will create a .pyc file in the __pycache__ folder.


