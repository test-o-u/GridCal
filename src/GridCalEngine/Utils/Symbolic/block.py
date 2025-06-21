# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Sequence, List
from GridCalEngine.Utils.Symbolic.symbolic import Var, Const, Expr, compile_numba_functions


@dataclass
class Block:
    """A declarative container: no methods, just equation lists."""

    algebraic_vars: List[Var]
    algebraic_eqs: List[Expr]
    state_vars: List[Var]
    state_eqs: List[Expr]
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.algebraic_vars) != len(self.algebraic_eqs):
            raise ValueError("algebraic_vars and algebraic_eqs must have the same length")
        if len(self.state_vars) != len(self.state_eqs):
            raise ValueError("state_vars and state_eqs must have the same length")


def compose_block(name: str,
                  inputs: list[Var],
                  outputs: list[Var],
                  inner_blocks: list[Block]) -> Block:
    """
    Bundle *inner_blocks* into a single reusable Block.

    *inputs*  – Vars that outside world will drive
    *outputs* – Vars that outside world will observe
    Anything else is considered internal wiring.
    """
    alg_v, alg_e, st_v, st_e = [], [], [], []

    for blk in inner_blocks:
        alg_v.extend(blk.algebraic_vars)
        alg_e.extend(blk.algebraic_eqs)
        st_v.extend(blk.state_vars)
        st_e.extend(blk.state_eqs)

    # Keep only those algebraic signals that the outer world needs to see
    # (outputs) or drive (inputs); everybody else is internal
    exposed = set(inputs) | set(outputs)
    keep = [i for i, v in enumerate(alg_v) if v in exposed]
    alg_v = [alg_v[i] for i in keep]
    alg_e = [alg_e[i] for i in keep]

    return Block(
        algebraic_vars=alg_v,
        algebraic_eqs=alg_e,
        state_vars=st_v,
        state_eqs=st_e,
        name=name,
    )


# --------------------------------------------------------------------------------------
# Block factory helpers – each returns (output_var, Block)
# --------------------------------------------------------------------------------------

def constant(value: float, name: str = "const") -> Tuple[Var, Block]:
    y = Var(name)
    blk = Block([y], [y - Const(value)], [], [])
    return y, blk


def gain(k: float, u: Var, name: str = "gain_out") -> Tuple[Var, Block]:
    y = Var(name)
    blk = Block([y], [y - Const(k) * u], [], [])
    return y, blk


def adder(inputs: Sequence[Var], name: str = "sum_out") -> Tuple[Var, Block]:
    if not inputs:
        raise ValueError("adder() needs at least one input variable")
    y = Var(name)
    expr: Expr = inputs[0]
    for v in inputs[1:]:
        expr = expr + v
    blk = Block([y], [y - expr], [], [])
    return y, blk


def integrator(u: Var, name: str = "x") -> Tuple[Var, Block]:
    x = Var(name)
    blk = Block([], [], [x], [u])
    return x, blk
