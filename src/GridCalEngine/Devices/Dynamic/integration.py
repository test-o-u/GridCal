import numpy as np
from scipy.sparse import bmat, identity
from scipy.sparse.linalg import spsolve

class Integration:
    """
    Base class for implicit iterative methods.
    """
    @staticmethod
    def calc_jac(dae, dt):
        pass
    
    @staticmethod
    def calc_q(x, f, Tf, h, x0, f0):
        pass
    
    @staticmethod
    def step(dae, dt, tol=1e-6, max_iter=10):
        """
        Perform an implicit integration step with Newton-Raphson.
        """
        x0, y0, f0 = dae.x.copy(), dae.y.copy(), dae.f.copy()
        
        for iteration in range(max_iter):
            jac = dae.method.calc_jac(dae, dt)
            qg = dae.method.calc_q(dae.x, dae.f, dae.Tf, dt, x0, f0)
            qg[dae.nx:] += dae.g  # Include algebraic residuals
            
            # Solve linear system
            inc = spsolve(jac, -qg)
            
            # Update variables
            dae.x += inc[:dae.nx]
            dae.y += inc[dae.nx:]
            
            # Recompute f and g
            dae.update_fg()
            
            # Check convergence
            if np.linalg.norm(inc, np.inf) < tol:
                return True
        
        # Restore previous values if not converged
        dae.x, dae.y, dae.f = x0, y0, f0
        return False

class BackEuler(Integration):
    """
    Backward Euler method.
    """
    @staticmethod
    def calc_jac(dae, dt):
        return bmat([[identity(dae.nx) - dt * dae.dfx, dae.dgx],
                     [-dt * dae.dfy, dae.dgy]], format='csr')
    
    @staticmethod
    def calc_q(x, f, Tf, dt, x0, f0):
        return Tf * (x - x0) - dt * f

class Trapezoid(Integration):
    """
    Trapezoidal integration method.
    """
    @staticmethod
    def calc_jac(dae, dt):
        return bmat([[identity(dae.nx) - 0.5 * dt * dae.dfx, dae.dgx],
                     [-0.5 * dt * dae.dfy, dae.dgy]], format='csr')
    
    @staticmethod
    def calc_q(x, f, Tf, dt, x0, f0):
        return Tf * (x - x0) - 0.5 * dt * (f + f0)

method_map = {
    "trapezoid": Trapezoid,
    "backeuler": BackEuler,
}
