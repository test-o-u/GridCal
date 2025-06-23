# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Tuple, Sequence, List
from GridCalEngine.Utils.Symbolic.symbolic import Var, Const, Expr


def _new_uid() -> int:
    """Generate a fresh UUID‑v4 string."""
    return uuid.uuid4().int


@dataclass(frozen=True)
class Block:
    """
    This represents a group of equations or a group of blocks
    """
    uid: int = field(default_factory=_new_uid)

    # internal vars
    algebraic_vars: List[Var] = field(default_factory=list)
    algebraic_eqs: List[Expr] = field(default_factory=list)
    state_vars: List[Var] = field(default_factory=list)
    state_eqs: List[Expr] = field(default_factory=list)

    name: str = ""

    # vars to make this recursive
    children: list["Block"] = field(default_factory=list)
    in_vars: List[Var] = field(default_factory=list)
    out_vars: List[Var] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.algebraic_vars) != len(self.algebraic_eqs):
            raise ValueError("algebraic_vars and algebraic_eqs must have the same length")
        if len(self.state_vars) != len(self.state_eqs):
            raise ValueError("state_vars and state_eqs must have the same length")

    def add(self, val: "Block"):
        """
        Add another block
        :param val: Block
        """
        self.children.append(val)

    def get_all_blocks(self) -> List[Block]:
        """
        Depth-first collection of all *primitive* Blocks.
        """

        flat: List[Block] = [self]
        for el in self.children:
            flat.extend(el.get_all_blocks())

        return flat

    def to_dict(self):
        return dict()

    def parse(self, data):
        pass


def compose_block(name: str,
                  inputs: list[Var],
                  outputs: list[Var],
                  inner_blocks: list[Block]) -> Block:
    """
    Bundle *inner_blocks* into a single reusable Block.
    :param name: Name for the new block
    :param inputs: Vars that outside world will drive
    :param outputs:  Vars that outside world will observe
    :param inner_blocks:
    :return: Single block
    """
    alg_v = list()
    alg_e = list()
    st_v = list()
    st_e = list()

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


# ----------------------------------------------------------------------------------------------------------------------
# Pre defined blocks
# ----------------------------------------------------------------------------------------------------------------------

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


def pi_controller(err: Var, kp: float, ki: float, name: str = "pi") -> Block:
    up, blk_kp = gain(kp, err, f"{name}_up")
    ie, blk_int = integrator(err, f"{name}_int")
    ui, blk_ki = gain(ki, ie, f"{name}_ui")
    u, blk_sum = adder([up, ui], f"{name}_u")
    return Block(name="",
                 children=[blk_kp, blk_int, blk_ki, blk_sum],
                 in_vars=[err],
                 out_vars=[u])
