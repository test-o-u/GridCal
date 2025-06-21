# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
from typing import List, Dict, Callable
from GridCalEngine.Utils.Symbolic.symbolic import Var, Const, Expr, compile_numba_functions


class Port:
    """
    A scalar signal line.
    """

    def __init__(self, owner: "Block", name: str):
        self.owner, self.name = owner, name
        self.value: float = 0.0
        self.connections: list["Port"] = []
        self.sym: Var | None = None  # filled later for equation extraction

    def connect(self, other: "Port"):
        self.connections.append(other)


class Block:
    """
    Base class—all concrete blocks inherit from this.
    """

    def __init__(self, name: str):
        self.name = name
        self.inputs: Dict[str, Port] = {}
        self.outputs: Dict[str, Port] = {}
        self.state_vars: list[str] = []  # names of internal state scalars

    # helpers ------------------------------------------------------
    def in_port(self, n: str) -> Port: return self.inputs[n]

    def out_port(self, n: str) -> Port: return self.outputs[n]

    # overridable --------------------------------------------------
    def step(self, dt: float, t: float):
        """Compute new outputs and update any internal state."""
        raise NotImplementedError

    def equations(self) -> list[Expr]:
        """Return symbolic equations for this block (default none)."""
        return []


def connect(src: Port, dst: Port):
    """
    Wiring helper
    :param src:
    :param dst:
    :return:
    """
    src.connect(dst)


class EquationBlock(Block):
    """
    Generic block defined only by symbolic equations—no subclassing required.

    Parameters
    ----------
    name        : str
    inputs      : list[Var]
    outputs     : dict[str, Expr] (Exprs may reference state vars)
    states      : dict[Var, Expr] | None
        Mapping from state variable to its *RHS* time derivative expression.
        Omit or pass {} for stateless blocks.
    integrator  : callable(x, rhs, dt) -> new_x
        Defaults to forward Euler.
    """

    # ---------- default forward-Euler integrator -----------------------
    @staticmethod
    def _euler(x: float, rhs: float, dt: float) -> float:
        return x + dt * rhs

    # ------------------------------------------------------------------
    def __init__(self,
                 name: str,
                 inputs: List[Var],
                 outputs: Dict[str, Expr],
                 states: Dict[Var, Expr] | None = None,
                 integrator: Callable[[float, float, float], float] | None = None):
        super().__init__(name)

        # create input & output ports
        self.inputs = {v.name: Port(self, v.name) for v in inputs}
        self.outputs = {k: Port(self, k) for k in outputs}

        self._state_vars: List[Var] = list(states or {})
        self.state_vars = [v.name for v in self._state_vars]  # Engine’s API

        # runtime storage for state values
        self._state_values = {v.uid: 0.0 for v in self._state_vars}

        # pick integrator
        self._step_state = integrator or self._euler

        # ---- compile kernels -----------------------------------------
        all_inputs_for_kernel = self._state_vars + list(self.inputs.values())

        # RHS kernel (only if dynamic)
        if states:
            rhs_exprs = [states[v] for v in self._state_vars]
            self._rhs_kernel = compile_numba_functions(rhs_exprs, sorting_vars=all_inputs_for_kernel)
        else:
            self._rhs_kernel = None

        # output kernel
        out_exprs = list(outputs.values())
        self._out_kernel = compile_numba_functions(out_exprs,
                                                   sorting_vars=self._state_vars + list(self.inputs.values()))

    # ------------------------------------------------------------------
    def step(self, dt: float, t: float):
        # positional list: states | inputs
        in_values = [self._state_values[v.uid] for v in self._state_vars] + [p.value for p in self.inputs]

        # 1. advance states if any
        if self._rhs_kernel:
            rhs_vals = self._rhs_kernel(*in_values)
            if len(self._state_vars) == 1:
                rhs_vals = (rhs_vals,)  # unify scalar/tuple
            for v, rhs in zip(self._state_vars, rhs_vals):
                self._state_values[v.uid] = self._step_state(self._state_values[v.uid],
                                                             rhs, dt)

            # refresh in_values with updated state values
            in_values = [self._state_values[v.uid] for v in self._state_vars] + in_values[len(self._state_vars):]

        # 2. compute outputs
        outs = self._out_kernel(*in_values)
        if len(self.outputs) == 1:
            outs = (outs,)
        for port, val in zip(self.outputs.values(), outs):
            port.value = val
