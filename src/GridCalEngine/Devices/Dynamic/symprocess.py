# import sympy as sp
# import os
# import py_compile
# import importlib.util

# def process_component(name, expressions):
#     """
#     Processes a component by parsing its string expressions, computing Jacobians,
#     lambdifying, and saving as a .py file which is then compiled to .pyc.
#     """
#     # Extract variable names
#     all_vars = sorted(set(var for expr in expressions for var in sp.sympify(expr).free_symbols), key=str)
#     symbols = {str(var): sp.Symbol(str(var)) for var in all_vars}
    
#     # Convert expressions to symbolic form
#     symbolic_exprs = [sp.sympify(expr, locals=symbols) for expr in expressions]
    
#     # Compute Jacobians (derivatives w.r.t all variables)
#     jacobians = [sp.Matrix([expr]).jacobian(list(symbols.values())) for expr in symbolic_exprs]
    
#     # Lambdify everything
#     lambdified_exprs = [sp.lambdify(list(symbols.values()), expr) for expr in symbolic_exprs]
#     lambdified_jacobians = [sp.lambdify(list(symbols.values()), jac) for jac in jacobians]
    
#     # Generate Python code file
#     py_filename = f"{name}.py"
#     with open(py_filename, "w") as f:
#         f.write("import numpy as np\n")
#         f.write("import sympy as sp\n")
        
#         # Write symbolic variables
#         f.write("# Define symbolic variables\n")
#         for var_name, var in symbols.items():
#             f.write(f"{var_name} = sp.Symbol('{var_name}')\n")
        
#         # Write expressions
#         f.write("\n# Expressions\n")
#         for i, expr in enumerate(symbolic_exprs):
#             f.write(f"expr_{i} = {expr}\n")
        
#         # Write Jacobians
#         f.write("\n# Jacobians\n")
#         for i, jac in enumerate(jacobians):
#             f.write(f"jacobian_{i} = {jac}\n")
        
#         # Write lambdified versions
#         f.write("\n# Lambdified functions\n")
#         f.write("import sympy.utilities.lambdify as lambdify\n")
#         f.write("import numpy as np\n")
        
#         f.write("lambdified_exprs = [\n")
#         for i in range(len(lambdified_exprs)):
#             f.write(f"    lambdify(({', '.join(map(str, symbols.keys()))}), expr_{i}),\n")
#         f.write("]\n")
        
#         f.write("lambdified_jacobians = [\n")
#         for i in range(len(lambdified_jacobians)):
#             f.write(f"    lambdify(({', '.join(map(str, symbols.keys()))}), jacobian_{i}),\n")
#         f.write("]\n")
    
#     # Compile to .pyc
#     py_compile.compile(py_filename)
#     print(f"Generated and compiled {py_filename} -> {py_filename}c")
    
# # Example usage
# components = {
#     "component1": ["x**2 + y", "sin(x) + cos(y)"],
#     "component2": ["exp(x) + log(y)", "x*y - y**2"]
# }

# for name, expressions in components.items():
#     process_component(name, expressions)
