# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations
import json
import math
import uuid
import numba as nb
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Dict, Mapping, Union, List

NUMBER = Union[int, float]


# -----------------------------------------------------------------------------
# UUID helper
# -----------------------------------------------------------------------------

def _new_uid() -> str:
    """Generate a fresh UUID‑v4 string."""
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _to_expr(val: Any) -> "Expr":
    if isinstance(val, Expr):
        return val
    if isinstance(val, (int, float)):
        return Const(val)
    raise TypeError(f"Cannot convert {val!r} to Expr")


def _var_name(sym: Var | str) -> str:
    return sym.name if isinstance(sym, Var) else sym


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------

class Expr:
    """Abstract base class for all expression nodes."""

    uid: str  # real dataclass field lives in subclasses

    # ------------------------------------------------------------------
    # Numeric evaluation
    # ------------------------------------------------------------------
    def eval(self, **bindings: float | int) -> float | int:  # pragma: no cover – abstract
        raise NotImplementedError

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:  # pragma: no cover – abstract
        raise NotImplementedError

    __call__ = eval  # allow f(x=…)

    # ------------------------------------------------------------------
    # Differentiation (higher‑order)
    # ------------------------------------------------------------------
    def diff(self, var: Var | str, order: int = 1) -> "Expr":
        if order < 0:
            raise ValueError("order must be ≥ 0")
        expr: Expr = self
        for _ in range(order):
            expr = expr._diff1(var).simplify()
        return expr

    def _diff1(self, var: Var | str) -> "Expr":  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Simplification & substitution (no‑ops by default)
    # ------------------------------------------------------------------
    def simplify(self) -> "Expr":
        return self

    def subs(self, mapping: Dict[Any, "Expr"]) -> "Expr":
        return mapping.get(self, self)

    # ------------------------------------------------------------------
    # JSON I/O helpers
    # ------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# Atomic nodes
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Const(Expr):
    value: NUMBER
    uid: str = field(default_factory=_new_uid, init=False)

    def eval(self, **bindings: NUMBER) -> NUMBER:
        return self.value

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:
        return self.value

    def _diff1(self, var: Var | str) -> "Expr":
        return Const(0)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Var(Expr):
    name: str
    uid: str = field(default_factory=_new_uid, init=False)

    def eval(self, **bindings: NUMBER) -> NUMBER:
        try:
            return bindings[self.name]
        except KeyError as exc:
            raise ValueError(f"No value for variable '{self.name}'.") from exc

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:
        try:
            return uid_bindings[self.uid]
        except KeyError as exc:
            raise ValueError(f"No value for uid '{self.uid}'.") from exc

    def _diff1(self, var: Var | str) -> Expr:
        return Const(1 if self.name == _var_name(var) else 0)

    def subs(self, mapping: Dict[Any, Expr]) -> Expr:
        if self in mapping:
            return mapping[self]
        if self.name in mapping:
            return mapping[self.name]
        return self

    def __str__(self) -> str:
        return self.name


# ----------------------------------------------------------------------------------------------------------------------
# Binary & unary operators
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    uid: str = field(default_factory=_new_uid, init=False)

    _impl: ClassVar[Mapping[str, Callable[[NUMBER, NUMBER], NUMBER]]] = MappingProxyType({
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "**": lambda a, b: a ** b,
    })

    def eval(self, **bindings: NUMBER) -> NUMBER:
        """
        Evaluation using names
        :param bindings:
        :return:
        """
        return self._impl[self.op](self.left.eval(**bindings), self.right.eval(**bindings))

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:
        """
        Evaluate using uuid's
        :param uid_bindings:
        :return:
        """
        return self._impl[self.op](self.left.eval_uid(uid_bindings), self.right.eval_uid(uid_bindings))

    def _diff1(self, var: Var | str) -> Expr:
        """
        Differentiation of this expression w.r.t var
        :param var: variable to differentiate with respect to
        :return: Expression
        """
        u, v = self.left, self.right
        du, dv = u._diff1(var), v._diff1(var)
        if self.op == "+":
            return du + dv
        if self.op == "-":
            return du - dv
        if self.op == "*":
            return du * v + u * dv
        if self.op == "/":
            return (du * v - u * dv) / (v ** Const(2))
        if self.op == "**":
            if isinstance(v, Const):
                # numeric exponent
                n = v.value
                return Const(n) * (u ** Const(n - 1)) * du
            else:
                # general exponent: u**v = exp(v*log u)
                return self * (dv * log(u) + du * v / u)
        raise ValueError("Unsupported operator for diff")

    # --- simplification ------------------------------------------------------
    def simplify(self) -> Expr:
        """
        Simplify expression
        :return: Simplified expression
        """
        l, r = self.left.simplify(), self.right.simplify()
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(self._impl[self.op](l.value, r.value))

        if self.op == "+":
            if isinstance(l, Const) and l.value == 0:
                return r
            if isinstance(r, Const) and r.value == 0:
                return l

        if self.op == "*":
            for a, b in ((l, r), (r, l)):
                if isinstance(a, Const):
                    if a.value == 0:
                        return Const(0)
                    if a.value == 1:
                        return b

        if self.op == "**" and isinstance(r, Const):
            if r.value == 1:
                return l
            if r.value == 0:
                return Const(1)

        return BinOp(self.op, l, r)

    def subs(self, mapping: Dict[Any, Expr]) -> Expr:
        """
        Substitution
        :param mapping: mapping of variables to expressions
        :return:
        """
        if self in mapping:
            return mapping[self]
        return BinOp(self.op, self.left.subs(mapping), self.right.subs(mapping))

    def __str__(self) -> str:
        return f"({self.left}) {self.op} ({self.right})"


@dataclass(frozen=True)
class UnOp(Expr):
    op: str
    operand: Expr
    uid: str = field(default_factory=_new_uid, init=False)

    def eval(self, **bindings: NUMBER) -> NUMBER:
        val = self.operand.eval(**bindings)
        return -val if self.op == "-" else math.nan

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:
        val = self.operand.eval_uid(uid_bindings)
        return -val if self.op == "-" else math.nan

    def _diff1(self, var: Var | str) -> "Expr":
        return -self.operand._diff1(var) if self.op == "-" else Const(float("nan"))

    def simplify(self) -> Expr:
        """

        :return:
        """
        opnd = self.operand.simplify()
        if isinstance(opnd, Const):
            return Const(-opnd.value)
        return UnOp(self.op, opnd)

    def subs(self, mapping: Dict[Any, Expr]) -> Expr:
        """

        :param mapping:
        :return:
        """
        if self in mapping:
            return mapping[self]
        return UnOp(self.op, self.operand.subs(mapping))

    def __str__(self) -> str:
        return f"{self.op}({self.operand})"


# ----------------------------------------------------------------------------------------------------------------------
# Functional nodes
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Func(Expr):
    name: str
    arg: Expr
    uid: str = field(default_factory=_new_uid, init=False)

    _impl: ClassVar[Mapping[str, Callable[[NUMBER], NUMBER]]] = MappingProxyType({
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
    })

    # --- evaluation ----------------------------------------------------------
    def eval(self, **bindings: NUMBER) -> NUMBER:
        return self._impl[self.name](self.arg.eval(**bindings))

    def eval_uid(self, uid_bindings: Dict[str, NUMBER]) -> NUMBER:
        return self._impl[self.name](self.arg.eval_uid(uid_bindings))

    # --- differentiation (chain rule) ---------------------------------------
    def _diff1(self, var: Var | str) -> "Expr":
        u = self.arg
        du = u._diff1(var)
        if isinstance(du, Const) and du.value == 0:
            return Const(0)
        if self.name == "sin":
            return cos(u) * du
        if self.name == "cos":
            return -sin(u) * du
        if self.name == "tan":
            return (sec(u) ** Const(2)) * du  # sec defined later
        if self.name == "exp":
            return exp(u) * du
        if self.name == "log":
            return du / u
        if self.name == "sqrt":
            return du / (Const(2) * sqrt(u))
        if self.name == "asin":
            return du / sqrt(Const(1) - u ** Const(2))
        if self.name == "acos":
            return -du / sqrt(Const(1) - u ** Const(2))
        if self.name == "atan":
            return du / (Const(1) + u ** Const(2))
        if self.name == "sinh":
            return cosh(u) * du
        if self.name == "cosh":
            return sinh(u) * du
        raise ValueError(f"Unknown function '{self.name}'")

    # --- simplification ------------------------------------------------------
    def simplify(self) -> "Expr":
        a = self.arg.simplify()
        if isinstance(a, Const):
            try:
                return Const(self._impl[self.name](a.value))
            except ValueError:
                pass  # domain error – keep symbolic
        return Func(self.name, a)

    def subs(self, mapping: Dict[Any, "Expr"]) -> "Expr":
        if self in mapping:
            return mapping[self]
        return Func(self.name, self.arg.subs(mapping))

    def __str__(self) -> str:
        return f"{self.name}({self.arg})"


# Helpers for functions not primitive nodes (sec for tan derivative)

def sec(x: Any) -> Expr:
    return Const(1) / cos(x)


# -----------------------------------------------------------------------------
# Public constructor helpers
# -----------------------------------------------------------------------------

def _make_unary(name: str):
    return lambda x: Func(name, _to_expr(x))


sin = _make_unary("sin")
cos = _make_unary("cos")
tan = _make_unary("tan")
exp = _make_unary("exp")
log = _make_unary("log")
sqrt = _make_unary("sqrt")
asin = _make_unary("asin")
acos = _make_unary("acos")
atan = _make_unary("atan")
sinh = _make_unary("sinh")
cosh = _make_unary("cosh")


# -----------------------------------------------------------------------------
# (De)serialisation helpers
# -----------------------------------------------------------------------------

def _expr_to_dict(expr: Expr) -> Dict[str, Any]:
    match expr:
        case Const(value=v, uid=uid):
            return {"type": "Const", "value": v, "uid": uid}
        case Var(name=n, uid=uid):
            return {"type": "Var", "name": n, "uid": uid}
        case BinOp(op=op, left=l, right=r, uid=uid):
            return {"type": "BinOp", "op": op, "left": _expr_to_dict(l), "right": _expr_to_dict(r), "uid": uid}
        case UnOp(op=op, operand=o, uid=uid):
            return {"type": "UnOp", "op": op, "operand": _expr_to_dict(o), "uid": uid}
        case Func(name=n, arg=a, uid=uid):
            return {"type": "Func", "name": n, "arg": _expr_to_dict(a), "uid": uid}
        case _:
            raise TypeError("Unsupported Expr subclass")


def _dict_to_expr(data: Dict[str, Any]) -> "Expr":
    t = data["type"]
    if t == "Const":
        obj: Expr = Const(data["value"])
    elif t == "Var":
        obj = Var(data["name"])
    elif t == "BinOp":
        obj = BinOp(data["op"], _dict_to_expr(data["left"]), _dict_to_expr(data["right"]))
    elif t == "UnOp":
        obj = UnOp(data["op"], _dict_to_expr(data["operand"]))
    elif t == "Func":
        obj = Func(data["name"], _dict_to_expr(data["arg"]))
    else:
        raise ValueError(f"Unknown type '{t}' in deserialisation")
    object.__setattr__(obj, "uid", data["uid"])
    return obj


# ----------------------------------------------------------------------------------------------------------------------
# Convenience top‑level helpers
# ----------------------------------------------------------------------------------------------------------------------

def diff(expr: Expr, var: Union[Var, str], order: int = 1) -> Expr:  # noqa: D401 – simple
    """Return ∂^order(expr)/∂var^order."""
    return expr.diff(var, order)


def eval_uid(expr: Expr, uid_bindings: Dict[str, NUMBER]) -> NUMBER:  # noqa: D401 – simple
    """Evaluate *expr* with a mapping from node UID → numeric value."""
    return expr.eval_uid(uid_bindings)


# ----------------------------------------------------------------------------------------------------------------------
# Collect variables in a deterministic order
# ----------------------------------------------------------------------------------------------------------------------
def _collect_vars(expr: Expr, out: List[Var]) -> None:
    """
    Depth-first, left-to-right variable harvest.
    :param expr: Some expression
    :param out: List to fill
    :return: None
    """
    if isinstance(expr, Var):
        if expr not in out:
            out.append(expr)
    elif isinstance(expr, BinOp):
        _collect_vars(expr.left, out)
        _collect_vars(expr.right, out)
    elif isinstance(expr, UnOp):
        _collect_vars(expr.operand, out)
    elif isinstance(expr, Func):
        _collect_vars(expr.arg, out)


def _all_vars(expr: Expr) -> List[Var]:
    """
    Collect all vars
    :param expr: Some expression
    :return: List of vars
    """
    lst: List[Var] = []
    _collect_vars(expr, lst)
    return lst


# ----------------------------------------------------------------------
# Emit a pure-Python (Numba-friendly) expression string
# ----------------------------------------------------------------------
def _emit(expr: Expr, uid_map: Dict[str, str]) -> str:
    if isinstance(expr, Const):
        return repr(expr.value)
    if isinstance(expr, Var):
        return uid_map[expr.uid]  # positional variable
    if isinstance(expr, UnOp):
        return f"-({_emit(expr.operand, uid_map)})"
    if isinstance(expr, BinOp):
        return f"({_emit(expr.left, uid_map)} {expr.op} {_emit(expr.right, uid_map)})"
    if isinstance(expr, Func):
        return f"math.{expr.name}({_emit(expr.arg, uid_map)})"
    raise TypeError(type(expr))


# ----------------------------------------------------------------------
# Public compiler
# ----------------------------------------------------------------------
def compile_numba(expr: Expr,
                  ordering: list[Union[Var, str]] | None = None,
                  generate_doc_string=True):
    """
    JIT-compile *expr* to a positional Numba function.
    :param expr: symbolic.Expr
    :param ordering: Optional list that fixes the argument order.
                     Items may be Var objects or variable names (str).
                     If None, variables are auto-collected left-to-right.
    :param generate_doc_string: Generate the docstring explaining the inputs
    :return:
    """

    # ---------- choose variable list ----------
    if ordering is None:
        vars_: list[Var] = []
        _collect_vars(expr, vars_)  # auto mode
    else:
        # normalise to Var objects
        vars0_: list[Var] = []
        _collect_vars(expr, vars0_)
        name_map = {v.name: v for v in vars0_}
        vars_ = [v if isinstance(v, Var) else name_map[v] for v in ordering]

    # Mapping: uid → positional symbol (v0, v1, …)
    uid_sym = {v.uid: f"v{i}" for i, v in enumerate(vars_)}

    # Build source
    body = _emit(expr, uid_sym)
    arg_list = ", ".join(uid_sym[v.uid] for v in vars_)
    src = f"def _f({arg_list}):\n    return {body}\n"

    ns = {"math": math}
    exec(src, ns)
    fn = ns["_f"]

    if generate_doc_string:
        fn.__doc__ = "Positional order: " + ", ".join(f"{sym} → {v.name}" for v, sym in zip(vars_, uid_sym.values()))

    return nb.njit(fn)  # JIT-compile and return


# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

__all__ = [
    "Expr", "Const", "Var", "BinOp", "UnOp", "Func",
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "asin", "acos", "atan", "sinh", "cosh",
    "diff", "eval_uid",
]
