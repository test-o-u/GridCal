# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Union

from GridCalEngine.Simulations.Dynamic_old.utils import json
from GridCalEngine.Utils.Symbolic.symbolic import Expr, _to_expr, BinOp, UnOp, _dict_to_expr, _expr_to_dict

NUMBER = Union[int, float]


def _new_uid() -> int:
    """Generate a fresh UUID‑v4 string."""
    return uuid.uuid4().int



class Expr:
    """
    Abstract base class for all expression nodes.
    """

    uid: str  # real dataclass field lives in subclasses

    def eval(self, **bindings: float | int) -> float | int:  # pragma: no cover – abstract
        """
        Numeric evaluation
        :param bindings:
        :return:
        """
        raise NotImplementedError

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:  # pragma: no cover – abstract
        """

        :param uid_bindings:
        :return:
        """
        raise NotImplementedError

    __call__ = eval  # allow f(x=…)

    def diff(self, var: Var | str, order: int = 1) -> "Expr":
        """
        Differentiation (higher‑order)
        :param var:
        :param order:
        :return:
        """
        if order < 0:
            raise ValueError("order must be >= 0")
        expr: Expr = self
        for _ in range(order):
            expr = expr._diff1(var).simplify()
        return expr

    def _diff1(self, var: Var | str) -> "Expr":  # pragma: no cover
        raise NotImplementedError

    def simplify(self) -> "Expr":
        """
        Simplification & substitution (no‑ops by default)
        :return:
        """
        return self

    def subs(self, mapping: Dict[Any, "Expr"]) -> "Expr":
        return mapping.get(self, self)

    def to_dict(self) -> Dict[str, Any]:
        return _expr_to_dict(self)

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **json_kwargs)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Expr":
        return _dict_to_expr(data)

    @staticmethod
    def from_json(blob: str) -> "Expr":
        return _dict_to_expr(json.loads(blob))

    # ------------------------------------------------------------------
    # Operator helpers
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> "Expr":
        return BinOp("+", self, _to_expr(other))

    def __radd__(self, other: Any) -> "Expr":
        return BinOp("+", _to_expr(other), self)

    def __sub__(self, other: Any) -> "Expr":
        return BinOp("-", self, _to_expr(other))

    def __rsub__(self, other: Any) -> "Expr":
        return BinOp("-", _to_expr(other), self)

    def __mul__(self, other: Any) -> "Expr":
        return BinOp("*", self, _to_expr(other))

    def __rmul__(self, other: Any) -> "Expr":
        return BinOp("*", _to_expr(other), self)

    def __truediv__(self, other: Any) -> "Expr":
        return BinOp("/", self, _to_expr(other))

    def __rtruediv__(self, other: Any) -> "Expr":
        return BinOp("/", _to_expr(other), self)

    def __pow__(self, other: Any) -> "Expr":
        return BinOp("**", self, _to_expr(other))

    def __rpow__(self, other: Any) -> "Expr":
        return BinOp("**", _to_expr(other), self)

    def __neg__(self) -> "Expr":
        return UnOp("-", self)

    def __str__(self) -> str:  # pragma: no cover – abstract
        """
        Display helper
        :return:
        """
        raise NotImplementedError

    __repr__ = __str__



@dataclass
class EventParam(Expr):
    value: float
    new_value: float
    time_step: int
    name: str
    uid: int = field(default_factory=_new_uid, init=False)

    def __post_init__(self):
        # "Freeze" name and uid after initialization by setting private flags
        object.__setattr__(self, "_frozen_name", self.name)
        object.__setattr__(self, "_frozen_uid", self.uid)

    def __setattr__(self, key, value):
        if hasattr(self, "_frozen_name") and key == "name":
            raise AttributeError("Cannot modify 'name' after initialization.")
        if hasattr(self, "_frozen_uid") and key == "uid":
            raise AttributeError("Cannot modify 'uid' after initialization.")
        super().__setattr__(key, value)

    def check_value(self, t):
        if t == self.time_step:
            return self.new_value
        else:
            return None

    def eval(self, **bindings: NUMBER) -> NUMBER:
        try:
            return bindings[self.name]
        except KeyError as exc:
            raise ValueError(f"No value for variable '{self.name}'.") from exc

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:
        return self.value

    def _diff1(self, EventParam: EventParam | str):
        return 0

    def subs(self, mapping: Dict[Any, Expr]) -> Expr:
        if self in mapping:
            return mapping[self]
        if self.name in mapping:
            return mapping[self.name]
        return self

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: "Var"):
        return self.uid == other.uid
