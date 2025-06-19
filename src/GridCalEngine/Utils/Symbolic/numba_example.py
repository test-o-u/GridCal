# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from symbolic import Var, sin, exp, compile_numba

x, y, x2 = Var("x"), Var("y"), Var("x")
expr = sin(x) * exp(y) + x2**2

f_fast = compile_numba(expr, ordering=[x, y, x2])

# Argument order is in the docstring:
# print(f_fast.__doc__)  → "Positional order: v0 → x, v1 → y, v2 → x"
print(f_fast(1.0, 2.0, 3.0))   # x=1, y=2, x2=3
