# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb
from typing import List, Dict, Any
from GridCalEngine.Devices.Parents.editable_device import EditableDevice, DeviceType
from GridCalEngine.Devices.Dynamic.variable import Var
from GridCalEngine.Devices.Dynamic.equation import Equation


class DynamicModel(EditableDevice):
    "This class contains the parameters and variables needed to construct a dynamic model"

    def __init__(self, name: str = "", idtag: str | None = None):
        """
        DynamicModel class constructor
        :param name: Name of the Model
        """
        super().__init__(name=name,
                         idtag=idtag,
                         code="",
                         device_type=DeviceType.DynModel, )

        self._algebraic_var_input: List[Var] = list()
        self._state_var_input: List[Var] = list()

        self._algebraic_var_output: List[Var] = list()
        self._state_var_output: List[Var] = list()

        self._algebraic_equations: List[Equation] = list()
        self._state_equations: List[Equation] = list()

    @property
    def n_input(self):
        return len(self._algebraic_var_input) + len(self._state_var_input)

    @property
    def n_output(self):
        return len(self._algebraic_var_output) + len(self._state_var_output)


class Sum(DynamicModel):

    def __init__(self, name: str, A: Var, B: Var):
        super().__init__(name, None)

        self.out = Var("C_" + name)
        eq = Equation(self.out, A.symbol + B.symbol)

        self._algebraic_var_input = [A, B]
        self._algebraic_var_output = [self.out]
        self._algebraic_equations = [eq]

if __name__ == "__main__":
    # model Suma 1
    # suma1 = DynamicModel(name="Suma 1")
    # A = Var("A")
    # B = Var("B")
    # C = Var("C")
    # eq1 = Equation(C, A.symbol + B.symbol)

    A_ = Var("A")
    B_ = Var("B")

    suma1 = Sum(name="Suma 1", A=A_, B=B_)

    suma2 = Sum(name="Suma 2", A=suma1.out, B=B_)

    print(suma1._algebraic_equations[0])
    print(suma2._algebraic_equations[0])
