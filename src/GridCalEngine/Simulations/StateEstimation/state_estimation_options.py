# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from GridCalEngine.enumerations import SolverType


class StateEstimationOptions:

    def __init__(self, solver: SolverType = SolverType.NR,
                 tol: float = 1e-9,
                 trust_radius: float = 1.0,
                 max_iter: int = 100, verbose: int = 0,
                 prefer_correct: bool = True, c_threshold: int = 4.0,
                 fixed_slack: bool = False):
        """
        StateEstimationOptions
        :param tol: Tolerance
        :param trust_radius: Solution trust radius (value around 1)
        :param max_iter: Maximum number of iterations
        :param verbose: Verbosity level (1 light, 2 heavy)
        :param prefer_correct: Prefer measurement correction? otherwise measurement deletion is used
        :param c_threshold: confidence threshold (default 4.0)
        :param fixed_slack: if true, the measurements on the slack bus are omitted
        """
        self.solver = solver
        self.tolerance = tol
        self.trust_radius = trust_radius
        self.max_iter = max_iter
        self.verbose = verbose
        self.prefer_correct = prefer_correct
        self.c_threshold = c_threshold
        self.fixed_slack: bool = fixed_slack
