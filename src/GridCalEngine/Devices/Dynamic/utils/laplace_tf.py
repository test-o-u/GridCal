import sympy as sp

class LaplaceTF:
    "This class contains the methods to convert a transfer function in the Laplace domain to the time domain."
    def __init__(self, tf_string, input='u', output='y'):
        self.tf_string = tf_string
        self.input = input
        self.output = output
        self.t = sp.Symbol('t')
        self.s = sp.Symbol('s')

    def _poly_to_time_expr(self, coeffs, var_func):
        expr = 0
        for i, coeff in enumerate(reversed(coeffs)):
            expr += coeff * sp.diff(var_func(self.t), self.t, i)
        return expr

    def process_tf(self):
        tf = sp.sympify(self.tf_string)
        num, den = sp.fraction(tf)

        num_poly = sp.Poly(num, self.s)
        den_poly = sp.Poly(den, self.s)

        u_func = sp.Function(self.input)
        y_func = sp.Function(self.output)

        lhs = self._poly_to_time_expr(den_poly.all_coeffs(), y_func)
        rhs = self._poly_to_time_expr(num_poly.all_coeffs(), u_func)

        eq = sp.Eq(lhs, rhs)

        # Identify highest derivative
        derivatives = sorted(lhs.atoms(sp.Derivative), key=lambda d: d.derivative_count, reverse=True)
        if not derivatives:
            raise ValueError("No derivatives found in TF output.")
        
        highest = derivatives[0]

        # Solve for that derivative and move time constant to the left side
        rearranged = sp.solve(eq, highest)[0]
        
        lhs_coeff = lhs.coeff(highest)
        rhs_cleaned = rearranged * lhs_coeff

        rhs_cleaned_simple = self.replace_func_with_symbols(rhs_cleaned)

        return rhs_cleaned_simple
    
    def replace_func_with_symbols(self, expr):
        subs_map = {}
        func_atoms = expr.atoms(sp.Function)
        
        for func in func_atoms:
            name = str(func.func)
            if func.args:
                subs_map[func] = sp.Symbol(name)
        
        # Handle Derivatives: y'(t), y''(t), etc.
        deriv_atoms = expr.atoms(sp.Derivative)
        for d in deriv_atoms:
            func_name = str(d.expr.func)
            order = d.derivative_count
            deriv_sym = sp.Symbol(f"{func_name}{order if order > 1 else ''}_dot")
            subs_map[d] = deriv_sym

        return expr.subs(subs_map)
        