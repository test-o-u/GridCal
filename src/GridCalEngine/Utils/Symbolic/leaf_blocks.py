
from GridCalEngine.Utils.Symbolic.symbolic import Var, Expr, compile_numba_functions
from GridCalEngine.Utils.Symbolic.block import Block, Port

class Constant(Block):
    def __init__(self, value: float, name="Const"):
        super().__init__(name)
        self.value = value
        self.outputs["out"] = Port(self, "out")

    def step(self, dt, t):
        self.outputs["out"].value = self.value


class Gain(Block):
    def __init__(self, k: float, name="Gain"):
        super().__init__(name)
        self.k = k
        self.inputs["in"]  = Port(self, "in")
        self.outputs["out"] = Port(self, "out")

    def step(self, dt, t):
        self.outputs["out"].value = self.k * self.inputs["in"].value


class Integrator(Block):
    """ẋ = u  (Euler discretisation)."""
    def __init__(self, name="Int"):
        super().__init__(name)
        self.inputs["in"]  = Port(self, "in")
        self.outputs["out"] = Port(self, "out")
        self.x = 0.0
        self.state_vars = ["x"]

    def step(self, dt, t):
        self.x += dt * self.inputs["in"].value
        self.outputs["out"].value = self.x


class SymbolicKernelBlock(Block):
    def __init__(self, name: str,
                 inputs: list[Var],
                 output_exprs: list[Expr]):
        super().__init__(name)
        # ports
        self.inputs  = {v.name: Port(self, v.name) for v in inputs}
        self.outputs = {f"y{i}": Port(self, f"y{i}") for i in range(len(output_exprs))}

        # compile
        self._order  = inputs
        self._orig_exprs = output_exprs
        self._kern   = compile_numba_functions(output_exprs, sorting_vars=inputs)

    def step(self, dt, t):
        args = [self.inputs[v.name].value for v in self._order]
        res  = self._kern(*args)
        if len(self.outputs) == 1:
            res = (res,)
        for val, port in zip(res, self.outputs.values()):
            port.value = val

    def equations(self):
        # yᵢ - f_i(...) == 0
        return [port.sym - expr for port, expr in zip(self.outputs.values(), self._orig_exprs)]