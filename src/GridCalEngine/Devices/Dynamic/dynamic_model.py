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

        self._algebraic_var_input: Dict[str, Var] = dict()
        self._state_var_input: Dict[str, Var] = dict()
        self._algebraic_var_output: Dict[str, Var] = dict()
        self._state_var_output: Dict[str, Var] = dict()
        self._algebraic_equations: Dict[str, Equation] = dict()
        self._state_equations: Dict[str, Equation] = dict()

    @property
    def n_input(self):
        return len(self._algebraic_var_input) + len(self._state_var_input)

    @property
    def n_output(self):
        return len(self._algebraic_var_output) + len(self._state_var_output)

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a dictionary representation of this object
        :return: Dictionary
        """
        return {
            "idtag": self.idtag,
            "algebraic_var_input": [e.to_dict() for _, e in self._algebraic_var_input.items()],
            "state_var_input": [e.to_dict() for _, e in self._state_var_input.items()],
            "algebraic_var_output": [e.to_dict() for _, e in self._algebraic_var_output.items()],
            "state_var_output": [e.to_dict() for _, e in self._state_var_output.items()],
            "algebraic_equations": [e.to_dict() for _, e in self._algebraic_equations.items()],
            "state_equations": [e.to_dict() for _, e in self._state_equations.items()],
        }

    def parse(self, data: Dict[str, List[Dict[str, Any]]] | List[int] | List[str]):
        """
        Parse the dictionary representation of this object
        :param data:
        :return:
        """
        self.name: str = data["name"]
        self.idtag: str = data["idtag"]

        self._algebraic_var_input.clear()
        for elm in data["algebraic_var_input"]:
            obj = Var(elm["name"])
            self.add_algebraic_var_input(obj)

        self._state_var_input.clear()
        for elm in data["state_var_input"]:
            obj = Var(elm["name"])
            self.add_state_var_input(obj)

        self._algebraic_var_output.clear()
        for elm in data["algebraic_var_output"]:
            obj = Var(elm["name"])
            self.add_algebraic_var_output(obj)

        self._state_var_output.clear()
        for elm in data["state_var_output"]:
            obj = Var(elm["name"])
            self.add_state_var_output(obj)

        self._algebraic_equations.clear()
        for elm in data["algebraic_equations"]:
            obj = Equation(elm["output"], elm["eq"])
            self.add_algebraic_equations(obj)

        self._state_equations.clear()
        for elm in data["state_equations"]:
            obj = Equation(elm["output"], elm["eq"])
            self.add_state_equations(obj)


    def add_algebraic_var_input(self, val: Var):
        self._algebraic_var_input[val.name] = val

    def add_state_var_input(self, val: Var):
        self._state_var_input[val.name] = val

    def add_algebraic_var_output(self, val: Var):
        self._algebraic_var_output[val.name] = val

    def add_state_var_output(self, val: Var):
        self._state_var_output[val.name] = val

    def add_algebraic_equations(self, val: Equation):
        self._algebraic_equations[val.output.name] = val

    def add_state_equations(self, val: Equation):
        self._state_equations[val.output.name] = val

    def get_algebraic_var_input(self, val: str):
        return self._algebraic_var_input[val]

    def get_state_var_input(self, val: str):
        return self._state_var_input[val]

    def get_algebraic_var_output(self, val: str):
        return self._algebraic_var_output[val]

    def get_state_var_output(self, val: str):
        return self._state_var_output[val]

    def get_algebraic_equations(self, val: str):
        return self._algebraic_var_output[val]

    def get_state_equations(self, val: str):
        return self._state_var_output[val]




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
