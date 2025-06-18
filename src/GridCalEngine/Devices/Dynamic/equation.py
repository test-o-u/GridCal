# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from uuid import uuid4
import sympy as smb
from GridCalEngine.Devices.Dynamic.variable import Var

class Equation:

    def __init__(self, output: Var, equation):
        self.idtag: int = uuid4().int

        self.output = output

        self.eq: str = smb.Eq(output.symbol, equation)

    def __str__(self):
        return smb.pretty(self.eq)