"""symbolic.py

Lightweight symbolic‑algebra framework with:
• Arithmetic operators
• Trigonometric & exponential functions
• Unique UUIDv4 identities for every node
• JSON (de)serialisation that preserves identities
• First‑order & partial derivatives (`Expr.diff` / `symbolic.diff`)

> **Fix 2025‑06‑19**  Annotated internal operator tables as **`ClassVar`** to
> prevent dataclass from treating them as mutable defaults. This resolves the
> `ValueError: mutable default <class 'mappingproxy'> for field _impl...` raised
> during import.

Example
~~~~~~~
```python
from symbolic import Var, sin, exp

x, y = Var('x'), Var('y')
f = sin(x) * exp(y) + x**3
print(f.diff('x'))  # cos(x)*exp(y) + 3*x**2
```
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Dict, Mapping, Union

Number = Union[int, float]

# -----------------------------------------------------------------------------
# UUID helper
# -----------------------------------------------------------------------------

def _new_uid() -> str:
    return str(uuid.uuid4())

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_expr(value: Any) -> "Expr":
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)):
        return Const(value)
    raise TypeError(f"Cannot convert {value!r} to Expr")


def _as_var_name(var: Union["Var", str]) -> str:
    return var.name if isinstance(var, Var) else var  # type: ignore[arg-type]

# -----------------------------------------------------------------------------
# Core expression hierarchy
# -----------------------------------------------------------------------------

class Expr:
    """Abstract base class for all expression nodes."""

    uid: str  # real field lives in subclasses

    # --- numeric evaluation --------------------------------------------------
    def eval(self, **bindings: Number) -> Number:  # pragma: no cover – abstract
        raise NotImplementedError

    # --- differentiation -----------------------------------------------------
    def diff(self, var: Union["Var", str]) -> "Expr":  # pragma: no cover
        raise NotImplementedError

    # allow f(x=…) shorthand
    __call__ = eval

    # --- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:  # noqa: D401 – simple
        return _expr_to_dict(self)

    def to_json(self, **json_kwargs: Any) -> str:  # noqa: D401 – simple
        return json.dumps(self.to_dict(), **json_kwargs)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Expr":
        return _dict_to_expr(data)

    @staticmethod
    def from_json(blob: str) -> "Expr":
        return _dict_to_expr(json.loads(blob))

    # --- operator overloading ------------------------------------------------
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

    # --- display -------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover – abstract
        raise NotImplementedError

    __repr__ = __str__

# -----------------------------------------------------------------------------
# Atomic expressions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Const(Expr):
    value: Number
    uid: str = field(default_factory=_new_uid, init=False)

    def eval(self, **bindings: Number) -> Number:
        return self.value

    def diff(self, var: Union["Var", str]) -> "Expr":
        return Const(0)

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Var(Expr):
    name: str
    uid: str = field(default_factory=_new_uid, init=False)

    def eval(self, **bindings: Number) -> Number:
        try:
            return bindings[self.name]
        except KeyError as exc:
            raise ValueError(f"No value provided for variable '{self.name}'.") from exc

    def diff(self, var: Union["Var", str]) -> "Expr":
        return Const(1 if self.name == _as_var_name(var) else 0)

    def __str__(self) -> str:
        return self.name

# -----------------------------------------------------------------------------
# Composite expressions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    uid: str = field(default_factory=_new_uid, init=False)

    _impl: ClassVar[Mapping[str, Callable[[Number, Number], Number]]] = MappingProxyType({
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "**": lambda a, b: a ** b,
    })

    def eval(self, **bindings: Number) -> Number:
        l, r = self.left.eval(**bindings), self.right.eval(**bindings)
        return self._impl[self.op](l, r)

    def diff(self, var: Union["Var", str]) -> "Expr":
        u, v = self.left, self.right
        du, dv = u.diff(var), v.diff(var)
        if self.op == "+":
            return du + dv
        if self.op == "-":
            return du - dv
        if self.op == "*":
            return du * v + u * dv  # product rule
        if self.op == "/":
            return (du * v - u * dv) / (v ** Const(2))  # quotient rule
        if self.op == "**":
            if isinstance(v, Const):
                n = v.value
                return Const(n) * (u ** Const(n - 1)) * du
            raise NotImplementedError("Derivative of u(x)**v(x) with non‑constant exponent not implemented.")
        raise ValueError(f"Unsupported binary operator '{self.op}'.")

    def __str__(self) -> str:
        return f"({self.left}) {self.op} ({self.right})"


@dataclass(frozen=True)
class UnOp(Expr):
    op: str
    operand: Expr
    uid: str = field(default_factory=_new_uid, init=False)

    def eval(self, **bindings: Number) -> Number:
        val = self.operand.eval(**bindings)
        if self.op == "-":
            return -val
        raise ValueError(f"Unsupported unary operator '{self.op}'.")

    def diff(self, var: Union["Var", str]) -> "Expr":
        if self.op == "-":
            return -self.operand.diff(var)
        raise ValueError(f"Unsupported unary operator '{self.op}'.")

    def __str__(self) -> str:
        return f"{self.op}({self.operand})"

# -----------------------------------------------------------------------------
# Functional expressions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Func(Expr):
    name: str
    arg: Expr
    uid: str = field(default_factory=_new_uid, init=False)

    _impl: ClassVar[Mapping[str, Callable[[Number], Number]]] = MappingProxyType({
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
    })

    def eval(self, **bindings: Number) -> Number:
        return self._impl[self.name](self.arg.eval(**bindings))

    def diff(self, var: Union["Var", str]) -> "Expr":
        inner = self.arg
        d_inner = inner.diff(var)
        if isinstance(d_inner, Const) and d_inner.value == 0:
            return d_inner
        if self.name == "sin":
            return cos(inner) * d_inner
        if self.name == "cos":
            return -sin(inner) * d_inner
        if self.name == "tan":
            return (Const(1) / (cos(inner) ** Const(2))) * d_inner
        if self.name == "exp":
            return exp(inner) * d_inner
        raise KeyError(f"Unknown function '{self.name}'.")

    def __str__(self) -> str:
        return f"{self.name}({self.arg})"

# -----------------------------------------------------------------------------
# Public helper constructors
# -----------------------------------------------------------------------------

def sin(x: Any) -> Expr:
    return Func("sin", _to_expr(x))

def cos(x: Any) -> Expr:
    return Func("cos", _to_expr(x))

def tan(x: Any) -> Expr:
    return Func("tan", _to_expr(x))

def exp(x: Any) -> Expr:
    return Func("exp", _to_expr(x))

# -----------------------------------------------------------------------------
# (De)serialisation utilities (unchanged)
# -----------------------------------------------------------------------------

def _expr_to_dict(expr: "Expr") -> Dict[str, Any]:
    match expr:
        case Const(value=value, uid=uid):
            return {"type": "Const", "uid": uid, "value": value}
        case Var(name=name, uid=uid):
            return {"type": "Var", "uid": uid, "name": name}
        case BinOp(op=op, left=left, right=right, uid=uid):
            return {"type": "BinOp", "uid": uid, "op": op, "left": _expr_to_dict(left), "right": _expr_to_dict(right)}
        case UnOp(op=op, operand=operand, uid=uid):
            return {"type": "UnOp", "uid": uid, "op": op, "operand": _expr_to_dict(operand)}
        case Func(name=name, arg=arg, uid=uid):
            return {"type": "Func", "uid": uid, "name": name, "arg": _expr_to_dict(arg)}
        case _:
            raise TypeError(f"Unsupported Expr subclass {type(expr).__name__}")


def _dict_to_expr(data: Dict[str, Any]) -> "Expr":
    t, uid = data["type"], data["uid"]
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
        raise ValueError(f"Unknown expression type '{t}' during deserialisation.")
    object.__setattr__(obj, "uid", uid)
    return obj

# -----------------------------------------------------------------------------
# Convenience diff wrapper
# -----------------------------------------------------------------------------

def diff(expr: Expr, var: Union[Var, str]) -> Expr:
    return expr.diff(var)

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------

__all__ = [
    "Expr",
    "Const",
    "Var",
    "BinOp",
    "UnOp",
    "Func",
    "sin",
    "cos",
    "tan",
    "exp",
    "diff",
]
