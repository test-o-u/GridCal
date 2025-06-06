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
from scipy import linalg


class Integration:
    """
    Base class for implicit iterative methods.
    """

    @staticmethod
    def calc_jac(dae, dt):
        """
        Calculates the Jacobian according to integration method.
        :param dae: DAE class
        :param dt: time interval
        :return:
        """
        pass

    @staticmethod
    def calc_f_res(x, f, Tf, h, x0, f0):
        """
        Calculates the state residual according to integration method.
        :param x: states variables values array
        :param f: states functions values array
        :param Tf: Tf
        :param h: integration step size
        :param x0: initial states variables values array
        :param f0: initial states functions values array
        :return:
        """
        pass

    @staticmethod
    def step(dae, dt, method, tol, max_iter):
        """
        Perform an implicit integration step with Newton-Raphson
        :param dae: DAE class
        :param dt: time interval
        :param method: integration method
        :param tol: tolerance
        :param max_iter: maximum of iterations
        :return:
        """
        x0, y0, f0 = dae.x.copy(), dae.y.copy(), dae.f.copy()

        for iteration in range(max_iter):
            # Compute Jacobian and residual
            jac = method.calc_jac(dae, dt)
            f_residual = method.calc_f_res(dae.x, dae.f, dae.Tf, dt, x0, f0)
            residual = np.vstack((f_residual.reshape(-1, 1), dae.g.reshape(-1, 1)))  # Include algebraic residuals

            # Solve linear system
            inc = spsolve(jac, -residual)

            # Update state and algebraic variables
            dae.x += 0.5 * inc[:dae.nx]
            dae.y += 0.5 * inc[dae.nx:]

            # Recompute f and g
            dae.update_fg()

            # Check convergence
            residual_error = np.linalg.norm(residual, np.inf)
            if residual_error < tol:
                return True

        # Restore previous values if not converged
        dae.x, dae.y, dae.f = x0, y0, f0
        return False

    @staticmethod
    def steadystate(dae, method, tol=1e-2, max_iter=10):
        """
        Perform an implicit integration step with Newton-Raphson.
        :param dae: DAE class
        :param method: iterative method
        :param tol: tolerance
        :param max_iter: maximum of iterations
        :return:
        """

        for iteration in range(max_iter):
            jac = method.calc_jac(dae)
            residual = np.vstack((dae.f.reshape(-1, 1), dae.g.reshape(-1, 1)))

            pdb.set_trace()


            # Solve linear system
            inc = spsolve(jac, -residual)

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
        """
        Builds Jacobian
        :param dae: DAE class
        :param dt: time interval
        :return: Jacobian in bmat format
        """
        return bmat([[identity(dae.nx) - dt * dae.dfx, -dt * dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')

    @staticmethod
    def calc_f_res(x, f, Tf, dt, x0, f0):
        """
        Calculates f resifuals
        :param x: states variables values array
        :param f: states functions values array
        :param Tf: Tf
        :param h: integration step size
        :param x0: initial states variables values array
        :param f0: initial states functions values array
        :return: f residuals array
        """
        return Tf @ (x - x0) - dt * f


class Trapezoid(Integration):
    """
    Trapezoidal integration method.
    """

    @staticmethod
    def calc_jac(dae, dt):
        """
        Builds Jacobian
        :param dae: DAE class
        :param dt: time interval
        :return: Jacobian in bmat format
        """
        return bmat([[identity(dae.nx) - 0.5 * dt * dae.dfx, -0.5 * dt * dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')

    @staticmethod
    def calc_f_res(x, f, Tf, dt, x0, f0):
        """
        Calculates f residuals
        :param x: states variables values array
        :param f: states functions values array
        :param Tf: Tf
        :param h: integration step size
        :param x0: initial states variables values array
        :param f0: initial states functions values array
        :return: f residuals array
        """
        return Tf @ (x - x0) - 0.5 * dt * (f + f0)


class SteadyState(Integration):
    """
    Steady-state computation.
    """

    @staticmethod
    def calc_jac(dae, dt=0.0):
        """
        Builds Jacobian
        :param dae: DAE class
        :param dt: time interval
        :return: Jacobian in bmat format
        """
        return bmat([[dae.dfx, dae.dfy],
                     [dae.dgx, dae.dgy]], format='csr')

    @staticmethod
    def calc_f_res(x, f, Tf, dt, x0, f0):
        pass


method_map = {
    "trapezoid": Trapezoid,
    "backeuler": BackEuler,
    "steadystate": SteadyState
}
