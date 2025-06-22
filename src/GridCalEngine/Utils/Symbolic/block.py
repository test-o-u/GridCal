# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Sequence, List
from GridCalEngine.Utils.Symbolic.symbolic import Var, Const, Expr


@dataclass(frozen=True)
class Block:
    """
    A block as a declarative container: no methods, just equation lists.
    """

    algebraic_vars: List[Var] = field(default_factory=list)
    algebraic_eqs: List[Expr] = field(default_factory=list)
    state_vars: List[Var] = field(default_factory=list)
    state_eqs: List[Expr] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.algebraic_vars) != len(self.algebraic_eqs):
            raise ValueError("algebraic_vars and algebraic_eqs must have the same length")
        if len(self.state_vars) != len(self.state_eqs):
            raise ValueError("state_vars and state_eqs must have the same length")


class BlockSystem:
    """
    A network of Blocks that behaves roughly like a Simulink diagram.
    """

    def __init__(self, blocks: Sequence[Block] | None = None):
        """
        Constructor
        :param blocks: list of blocks (optional)
        """

        if blocks is not None:
            self._blocks = list(blocks)
        else:
            self._blocks = list()

    @property
    def blocks(self):
        return self._blocks


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
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - Const(value)])
    return y, blk


def gain(k: float, u: Var | Const, name: str = "gain_out") -> Tuple[Var, Block]:
    y = Var(name)
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - Const(k) * u])
    return y, blk


def adder(inputs: Sequence[Var | Const], name: str = "sum_out") -> Tuple[Var, Block]:
    if len(inputs) == 0:
        raise ValueError("adder() needs at least one input variable")
    y = Var(name)
    expr: Expr = inputs[0]
    for v in inputs[1:]:
        expr += v
    blk = Block(algebraic_vars=[y], algebraic_eqs=[y - expr])
    return y, blk


def integrator(u: Var | Const, name: str = "x") -> Tuple[Var, Block]:
    x = Var(name)
    blk = Block(state_vars=[x], state_eqs=[u])
    return x, blk


def pi_controller(err: Var, kp: float, ki: float, name: str = "pi") -> Tuple[Var, List[Block]]:
    up, blk_kp = gain(kp, err, f"{name}_up")
    ie, blk_int = integrator(err, f"{name}_int")
    ui, blk_ki = gain(ki, ie, f"{name}_ui")
    u, blk_sum = adder([up, ui], f"{name}_u")
    return u, [blk_kp, blk_int, blk_ki, blk_sum]
