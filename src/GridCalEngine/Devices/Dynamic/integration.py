import numpy as np
import pdb
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
    def step(dae, dt, method, tol=1e-6, max_iter=10):
        """
        Perform an implicit integration step with Newton-Raphson.
        """
        x0, y0, f0 = dae.x.copy(), dae.y.copy(), dae.f.copy()
        
        for iteration in range(max_iter):
            jac = method.calc_jac(dae, dt)
            f_residual = method.calc_q(dae.x, dae.f, dae.Tf, dt, x0, f0)
            residual = np.vstack((f_residual.reshape(-1, 1), dae.g.reshape(-1, 1)))  # Include algebraic residuals

            # Solve linear system
            inc = spsolve(jac, -residual)
            
            # Update variables
            dae.x += inc[:dae.nx]
            dae.y += inc[dae.nx:]

            pdb.set_trace()
          
            #dae.concatenate()
            # Recompute f and g
            dae.update_fg()
            

            print(residual)

            # Check convergence
            residual_error = np.linalg.norm(residual, np.inf)
            if residual_error < tol:
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
        return bmat([[identity(dae.nx) - dt * dae.dfx, -dt * dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')
    
    @staticmethod
    def calc_q(x, f, Tf, dt, x0, f0):
        return Tf @ (x - x0) - dt * f

class Trapezoid(Integration):
    """
    Trapezoidal integration method.
    """
    @staticmethod
    def calc_jac(dae, dt):
        return bmat([[identity(dae.nx) - 0.5 * dt * dae.dfx, -0.5 * dt * dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')
    
    @staticmethod
    def calc_q(x, f, Tf, dt, x0, f0):
        return Tf @ (x - x0) - 0.5 * dt * (f + f0)

method_map = {
    "trapezoid": Trapezoid,
    "backeuler": BackEuler,
}
