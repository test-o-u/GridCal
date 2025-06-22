# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence
import math
import numpy as np
from typing import Dict, List, Literal
from matplotlib import pyplot as plt
from GridCalEngine.Utils.Symbolic.symbolic import Var, Expr, Const, BinOp, compile_numba_functions
from GridCalEngine.Utils.Symbolic.block import Block


class BlockSystem:
    """A network of Blocks that behaves roughly like a Simulink diagram."""

    # ------------------------------------------ helper: deep substitution until fixed‑point
    @staticmethod
    def _fully_substitute(expr: Expr, mapping: Dict[Var, Expr], max_iter: int = 10) -> Expr:
        cur = expr
        for _ in range(max_iter):
            nxt = cur.subs(mapping).simplify()
            if str(nxt) == str(cur):  # no further change
                break
            cur = nxt
        return cur

    def __init__(self, blocks: Sequence[Block] | None = None):
        """
        Constructor        
        :param blocks: list of blocks 
        """

        # Flatten the block lists, preserving declaration order
        self._algebraic_vars: List[Var] = list()
        self._algebraic_eqs: List[Expr] = list()
        self._state_vars: List[Var] = list()
        self._state_eqs: List[Expr] = list()

        self._alg_subs: Dict[Var, Expr] = dict()
        self._state_rhs: List[Expr] = list()
        self._rhs_fn = None

        if blocks is not None:
            self._blocks = list(blocks)
            self._initialize()
        else:
            self._blocks = list()
    
    def _initialize(self):

        # Flatten the block lists, preserving declaration order
        self._algebraic_vars.clear()
        self._algebraic_eqs.clear()
        self._state_vars.clear()
        self._state_eqs.clear()
        for b in self._blocks:
            self._algebraic_vars.extend(b.algebraic_vars)
            self._algebraic_eqs.extend(b.algebraic_eqs)
            self._state_vars.extend(b.state_vars)
            self._state_eqs.extend(b.state_eqs)

        # ---------------------------------- algebraic substitution map  y → rhs
        self._alg_subs: Dict[Var, Expr] = {}
        for y, eq in zip(self._algebraic_vars, self._algebraic_eqs):
            if isinstance(eq, BinOp) and eq.op == "-" and str(eq.left) == str(y):
                rhs = eq.right
            else:
                rhs = y - eq  # generic fallback

            # Flatten RHS using *already‑known* substitutions
            rhs_flat = self._fully_substitute(rhs, self._alg_subs)
            self._alg_subs[y] = rhs_flat

        # ---------------------------------- pure‑state RHS after full substitution
        self._state_rhs: List[Expr] = [
            self._fully_substitute(expr, self._alg_subs).simplify() for expr in self._state_eqs
        ]

        # ---------------------------------- JIT compile (if there are states)
        self._rhs_fn = None
        if self._state_vars:
            self._rhs_fn = compile_numba_functions(self._state_rhs, sorting_vars=self._state_vars)

    def rhs(self, state: Sequence[float]) -> np.ndarray:
        """Return 𝑑x/dt given the current *state* vector."""
        if self._rhs_fn is None:
            return np.array([])
        return np.asarray(self._rhs_fn(*state))

    def equations(self) -> Tuple[List[Expr], List[Expr]]:
        """(algebraic_eqs, state_eqs) as *originally declared* (no substitution)."""
        return list(self._algebraic_eqs), list(self._state_eqs)

    def simulate(
            self,
            t0: float,
            t_end: float,
            h: float,
            init_state: Sequence[float],
            *,
            method: Literal["rk4", "euler", "adaptive"] = "rk4",
            abs_tol: float = 1e-6,
            rel_tol: float = 1e-3,
            h_min: float | None = None,
            h_max: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the system.

        Parameters
        ----------
        t0, t_end : float
            Start / end times (same units as RHS)
        h : float
            Initial step size (for adaptive) or fixed size
        init_state : Sequence[float]
            Initial conditions (len == number of state variables)
        method : {"rk4", "euler", "adaptive"}
            Integration scheme
        abs_tol, rel_tol : float
            Error tolerances for adaptive mode (ignored otherwise)
        h_min, h_max : float | None
            Optional bounds for step size in adaptive mode
        """
        if len(self._state_vars) + len(self._algebraic_vars) == 0:
            self._initialize()

        if len(init_state) != len(self._state_vars):
            raise ValueError("init_state length mismatch with state_vars")

        if method == "adaptive":
            return self._simulate_adaptive(t0, t_end, h, init_state, abs_tol, rel_tol, h_min, h_max)
        else:
            return self._simulate_fixed(t0, t_end, h, init_state, method)

    # ------------------------------------------------------------------
    # Fixed‑step helpers (Euler, RK‑4)
    # ------------------------------------------------------------------
    def _simulate_fixed(
            self,
            t0: float,
            t_end: float,
            h: float,
            init_state: Sequence[float],
            method: Literal["rk4", "euler"],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(math.floor((t_end - t0) / h)) + 1
        t = np.linspace(t0, t0 + h * (n_steps - 1), n_steps)
        y = np.empty((n_steps, len(init_state)))
        y[0] = init_state

        rhs = self._rhs_fn  # local alias for speed

        for i in range(1, n_steps):
            yi = y[i - 1]
            if method == "euler":
                k1 = rhs(*yi)
                y[i] = yi + h * np.asarray(k1)
            elif method == "rk4":
                k1 = np.asarray(rhs(*yi))
                k2 = np.asarray(rhs(*(yi + 0.5 * h * k1)))
                k3 = np.asarray(rhs(*(yi + 0.5 * h * k2)))
                k4 = np.asarray(rhs(*(yi + h * k3)))
                y[i] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, y

    # ------------------------------------------------------------------
    # Adaptive RKF‑45 implementation
    # ------------------------------------------------------------------
    def _simulate_adaptive(
            self,
            t0: float,
            t_end: float,
            h0: float,
            init_state: Sequence[float],
            abs_tol: float,
            rel_tol: float,
            h_min: float | None,
            h_max: float | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rhs = self._rhs_fn
        t_list: List[float] = [t0]
        y_list: List[np.ndarray] = [np.asarray(init_state, dtype=float)]

        t = t0
        y = np.asarray(init_state, dtype=float)
        h = h0
        safety = 0.9
        pow_ = 0.2  # 1/(order+1) with order=4
        h_min = h_min if h_min is not None else h0 * 1e-6
        h_max = h_max if h_max is not None else (t_end - t0)

        while t < t_end:
            if h < h_min:
                raise RuntimeError("Step size underflow in adaptive integrator")
            if t + h > t_end:
                h = t_end - t  # final partial step

            # --------------------------------------------------
            # RKF‑45 coefficients
            # --------------------------------------------------
            k1 = np.asarray(rhs(*y))
            k2 = np.asarray(rhs(*(y + h * 0.25 * k1)))
            k3 = np.asarray(rhs(*(y + h * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2))))
            k4 = np.asarray(rhs(*(y + h * (1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3))))
            k5 = np.asarray(rhs(*(y + h * (439.0 / 216.0 * k1 - 8.0 * k2 + 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4))))
            k6 = np.asarray(rhs(*(y + h * (
                        -8.0 / 27.0 * k1 + 2.0 * k2 - 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 - 11.0 / 40.0 * k5))))

            y4 = y + h * (25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5)
            y5 = y + h * (
                    16.0 / 135.0 * k1 + 6656.0 / 12825.0 * k3 + 28561.0 / 56430.0 * k4
                    - 9.0 / 50.0 * k5 + 2.0 / 55.0 * k6
            )

            # Error estimate
            scale = abs_tol + np.maximum(np.abs(y), np.abs(y5)) * rel_tol
            err_est = np.max(np.abs(y5 - y4) / scale)

            if err_est <= 1.0:
                # Accept step
                t += h
                y = y5
                t_list.append(t)
                y_list.append(y)

            # Step size adjustment
            if err_est == 0.0:
                h_new = h_max
            else:
                h_new = safety * h * err_est ** (-pow_)
            h = min(max(h_new, h_min), h_max)

        t_arr = np.asarray(t_list)
        y_arr = np.vstack(y_list)
        return t_arr, y_arr
