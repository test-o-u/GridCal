# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pdb
import sys
import numpy as np
import scipy as sp
from scipy.sparse import bmat, identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Integration:
    """
    Base class for implicit iterative methods.
    """
    @staticmethod
    def calc_jac(dae, dt):
        """
        Calculates the Jacobian according to integration method.
        """
        pass
    
    
    @staticmethod
    def calc_f_res(x, f, Tconst, h, x0, f0):
        """
        Calculates the state residual according to integration method.
        """
        pass
    

    @staticmethod
    def step(dae, dt, method, tol, max_iter):
        """
        Perform an implicit integration step with Newton-Raphson and real-time plotting.
        """
        x0, y0, f0 = dae.x.copy(), dae.y.copy(), dae.f.copy()

        # Setup plotting
        residual_history = []
        dx_history = []
        dy_history = []

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        res_line, = ax[0].semilogy([], [], label='‖residual‖∞')
        dx_line, = ax[1].plot([], [], label='‖Δx‖∞')
        dy_line, = ax[1].plot([], [], label='‖Δy‖∞')

        ax[0].set_title("Residual")
        ax[1].set_title("Increments")
        for a in ax:
            a.grid(True)
            a.legend()

        plt.ion()
        plt.show()

        try:
            for iteration in range(max_iter):
                jac = method.calc_jac(dae, dt)

                # Jacobian singularity insight
                try:
                    cond = np.linalg.cond(jac.toarray())
                    if cond > 1e12:
                        print(f"[Warning] Iter {iteration}: Jacobian is nearly singular! cond ≈ {cond:.2e}")
                except Exception:
                    print(f"[Warning] Iter {iteration}: Jacobian too ill-conditioned to compute condition number.")

                f_residual = method.calc_f_res(dae.x, dae.f, dae.Tconst, dt, x0, f0)
                residual = np.vstack((f_residual.reshape(-1, 1), dae.g.reshape(-1, 1)))

                try:
                    inc = spsolve(jac, -residual)
                except Exception as e:
                    print(f"Linear solver failed at iteration {iteration}: {e}")
                    break

                dae.x += 0.5 * inc[:dae.nx]
                dae.y += 0.5 * inc[dae.nx:]

                dae.update_fg()

                residual_error = np.linalg.norm(residual, np.inf)
                dx_norm = np.linalg.norm(inc[:dae.nx], np.inf)
                dy_norm = np.linalg.norm(inc[dae.nx:], np.inf)

                # Save for plotting
                residual_history.append(residual_error)
                dx_history.append(dx_norm)
                dy_history.append(dy_norm)

                # Update plot
                x_vals = list(range(len(residual_history)))
                res_line.set_data(x_vals, residual_history)
                dx_line.set_data(x_vals, dx_history)
                dy_line.set_data(x_vals, dy_history)

                for a in ax:
                    a.relim()
                    a.autoscale_view()

                plt.pause(0.01)

                if residual_error < tol:
                    plt.ioff()
                    plt.close()
                    return True

        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Newton-Raphson stopped manually at iteration", iteration)
            print(f"‖residual‖∞ = {residual_error:.2e}, ‖Δx‖∞ = {dx_norm:.2e}, ‖Δy‖∞ = {dy_norm:.2e}")
            plt.ioff()
            plt.close()
            return False

        # Restore state if not converged
        dae.x, dae.y, dae.f = x0, y0, f0
        plt.ioff()
        plt.close()
        return False



    

    @staticmethod
    def steadystate(dae, method, tol=1e-2, max_iter=10):
        """
        Perform an implicit integration step with Newton-Raphson.
        """

        for iteration in range(max_iter):
            jac = method.calc_jac(dae)
            residual = np.vstack((dae.f.reshape(-1, 1), dae.g.reshape(-1, 1)))

            det = sp.linalg.det(jac.todense())

            # Solve linear system
            inc = spsolve(jac, -residual)

            #print(residual[14])
            print(dae.y[12])
            # pdb.set_trace()

            # Update variables
            dae.x += 0.5 * inc[:dae.nx]
            dae.y += 0.5 * inc[dae.nx:]

            # Recompute f and g
            dae.update_fg()
            np.set_printoptions(threshold=sys.maxsize)

           # Check convergence
            residual_error = np.linalg.norm(residual, np.inf)
            if residual_error < tol:
                return True
        
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
    def calc_f_res(x, f, Tconst, dt, x0, f0):
        return Tconst @ (x - x0) - dt * f


class Trapezoid(Integration):
    """
    Trapezoidal integration method.
    """
    @staticmethod
    def calc_jac(dae, dt):
        return bmat([[identity(dae.nx) - 0.5 * dt * dae.dfx, -0.5 * dt * dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')
    
    @staticmethod
    def calc_f_res(x, f, Tconst, dt, x0, f0):
        return Tconst @ (x - x0) - 0.5 * dt * (f + f0)


class SteadyState(Integration):
    """
    Steady-state computation.
    """
    @staticmethod
    def calc_jac(dae, dt=0.0):
        return bmat([[dae.dfx, dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')
    
    @staticmethod
    def calc_f_res(x, f, Tconst, dt, x0, f0):
        pass


method_map = {
    "trapezoid": Trapezoid,
    "backeuler": BackEuler,
    "steadystate": SteadyState
}
