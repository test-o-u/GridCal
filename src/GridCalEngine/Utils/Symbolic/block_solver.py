# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations
from typing import Tuple, Sequence
import numpy as np
from scipy.sparse import csc_matrix
from typing import Dict, List, Literal
from GridCalEngine.Utils.Symbolic.symbolic import Var, Expr, compile_numba_functions, get_jacobian, BinOp
from GridCalEngine.Utils.Symbolic.block import BlockSystem


def _fully_substitute(expr: Expr, mapping: Dict[Var, Expr], max_iter: int = 10) -> Expr:
    cur = expr
    for _ in range(max_iter):
        nxt = cur.subs(mapping).simplify()
        if str(nxt) == str(cur):  # no further change
            break
        cur = nxt
    return cur


class BlockSolver:
    """
    A network of Blocks that behaves roughly like a Simulink diagram.
    """

    def __init__(self, block_system: BlockSystem):
        """
        Constructor        
        :param block_system: BlockSystem
        """
        self.block_system = block_system

        # Flatten the block lists, preserving declaration order
        self._algebraic_vars: List[Var] = list()
        self._algebraic_eqs: List[Expr] = list()
        self._state_vars: List[Var] = list()
        self._state_eqs: List[Expr] = list()

        self._alg_subs: Dict[Var, Expr] = dict()
        self._state_rhs: List[Expr] = list()
        self._rhs_fn = None
        self._jac_fn = None
        self._n_state = 0

        self._initialize()

    def _initialize(self):
        """
        Initialize for simulation
        """
        # Flatten the block lists, preserving declaration order
        self._algebraic_vars.clear()
        self._algebraic_eqs.clear()
        self._state_vars.clear()
        self._state_eqs.clear()
        for b in self.block_system.get_flattened_blocks():
            self._algebraic_vars.extend(b.algebraic_vars)
            self._algebraic_eqs.extend(b.algebraic_eqs)
            self._state_vars.extend(b.state_vars)
            self._state_eqs.extend(b.state_eqs)

        # Substitute algebraic equations into state equations (fixed‑point)
        subst_map = dict()
        for v, eq in zip(self._algebraic_vars, self._algebraic_eqs):
            # assume the algebraic eq is either (v - rhs) or (rhs - v)
            if isinstance(eq, BinOp) and eq.op == "-":
                if eq.left == v:
                    subst_map[v] = eq.right  # v  – rhs = 0  → v = rhs
                elif eq.right == v:
                    subst_map[v] = eq.left  # rhs – v  = 0  → v = rhs

        self._state_eqs = [_fully_substitute(e, subst_map) for e in self._state_eqs]

        # Compile RHS and Jacobian
        self._rhs_fn = compile_numba_functions(self._state_eqs, sorting_vars=self._state_vars)
        self._jac_fn, _ = get_jacobian(self._state_eqs, self._state_vars)
        self._n_state = len(self._state_vars)

    def rhs(self, state: Sequence[float]) -> np.ndarray:
        """
        Return 𝑑x/dt given the current *state* vector.
        :param state: get the right-hand-side give a state vector
        """
        if self._rhs_fn is None:
            return np.array([])
        return np.asarray(self._rhs_fn(*state))

    def get_dummy_x0(self):
        return np.zeros(self._n_state)

    def equations(self) -> Tuple[List[Expr], List[Expr]]:
        """
        Return (algebraic_eqs, state_eqs) as *originally declared* (no substitution).
        """
        return self._algebraic_eqs, self._state_eqs

    def simulate(
            self,
            t0: float,
            t_end: float,
            h: float,
            x0: np.ndarray,
            method: Literal["rk4", "euler", "implicit_euler"] = "rk4",
            newton_tol: float = 1e-8,
            newton_max_iter: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param t0: start time
        :param t_end: end time
        :param h: step
        :param x0: initial values
        :param method: method
        :param newton_tol:
        :param newton_max_iter:
        :return: 1D time array, 2D array of simulated variables
        """
        if method == "euler":
            return self._simulate_fixed(t0, t_end, h, x0, stepper="euler")
        if method == "rk4":
            return self._simulate_fixed(t0, t_end, h, x0, stepper="rk4")
        if method == "implicit_euler":
            return self._simulate_implicit_euler(
                t0, t_end, h, x0,
                tol=newton_tol, max_iter=newton_max_iter,
            )
        raise ValueError(f"Unknown method '{method}'")

    def _simulate_fixed(self, t0, t_end, h, x0, stepper="euler"):
        """
        Fixed‑step helpers (Euler, RK‑4)
        :param t0:
        :param t_end:
        :param h:
        :param x0:
        :param stepper:
        :return:
        """
        steps = int(np.ceil((t_end - t0) / h))
        t = np.empty(steps + 1)
        y = np.empty((steps + 1, self._n_state))
        t[0] = t0
        y[0, :] = x0.copy()

        for i in range(steps):
            tn = t[i]
            xn = y[i]
            if stepper == "euler":
                k1 = np.array(self._rhs_fn(*xn))
                y[i + 1] = xn + h * k1
            elif stepper == "rk4":
                k1 = np.array(self._rhs_fn(*xn))
                k2 = np.array(self._rhs_fn(*(xn + 0.5 * h * k1)))
                k3 = np.array(self._rhs_fn(*(xn + 0.5 * h * k2)))
                k4 = np.array(self._rhs_fn(*(xn + h * k3)))
                y[i + 1] = xn + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise RuntimeError("unknown stepper")
            t[i + 1] = tn + h
        return t, y

    def _simulate_implicit_euler(self, t0, t_end, h, x0, tol=1e-8, max_iter=20):
        """

        :param t0:
        :param t_end:
        :param h:
        :param x0:
        :param tol:
        :param max_iter:
        :return:
        """
        steps = int(np.ceil((t_end - t0) / h))
        t = np.empty(steps + 1)
        y = np.empty((steps + 1, self._n_state))
        t[0] = t0
        y[0] = x0.copy()
        I = np.eye(self._n_state)

        for i in range(steps):
            xn = y[i]
            x_new = xn.copy()  # initial guess
            for _ in range(max_iter):
                f_val = np.array(self._rhs_fn(*x_new))
                res = x_new - xn - h * f_val
                if np.linalg.norm(res, np.inf) < tol:
                    break
                Jf = self._jac_fn(x_new)  # sparse matrix
                A = I - h * Jf.toarray()
                delta = np.linalg.solve(A, -res)
                x_new += delta
            else:
                raise RuntimeError("Newton failed in implicit Euler")
            y[i + 1] = x_new
            t[i + 1] = t[i] + h
        return t, y

    # ------------------------------------------------------------------ jacobian accessor
    def jacobian(self, x_vec: np.ndarray) -> csc_matrix:
        """

        :param x_vec:
        :return:
        """
        return self._jac_fn(*x_vec)

    def build_init_vector(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching the mapping
        """
        return np.array([mapping.get(v, 0.0) for v in self._state_vars], dtype=float)
