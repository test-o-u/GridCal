# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Dict
from uuid import uuid4
import sympy as smb
from sympy.parsing.sympy_parser import parse_expr
from GridCalEngine.Devices.Dynamic.variable import Var


class Equation:

    def __init__(self, output: Var, equation: smb.core.AtomicExpr):
        """

        :param output:
        :param equation:
        """
        self.idtag: int = uuid4().int

        self.output = output

        self.equation: smb.core.AtomicExpr = equation

        self.equality: smb.core.relational.Equality = smb.Eq(output.symbol, equation)

    def __str__(self):
        return smb.pretty(self.equality)


    def to_dict(self) -> Dict[str, str | int]:

        return {
            "idtag": self.idtag,
            "output": self.output.idtag,
            "eq": smb.srepr(self.equation)
        }

    def parse(self, data: Dict[str, str | int], var_dict: Dict[int, Var]):

        self.idtag = data["idtag"]
        self.output = var_dict.get(data["output"], None)
        if self.output is None:
            raise ValueError("Output var not found")

        equation = parse_expr(data["eq"])
        self.equation = equation

