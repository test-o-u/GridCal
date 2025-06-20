# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb
from typing import List

from GridCalEngine.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Devices.Dynamic.equation_2 import Equation


def get_equations(grid: MultiCircuit):
    # for e, elm in enumerate(grid.get_buses()):
    state_eqs: List[Equation] = list()
    algeb_eqs: List[Equation] = list()

    for e, elm in enumerate(grid.get_injection_devices_iter()):
        # rms model
        model = elm.rms_model.model

        # create dict

        model_dict = model.to_dict()

        # parse algebraic equations
        algeb_vars_output = model_dict["algebraic_var_output"]
        algeb_vars_uid = [var["uid"] for var in  algeb_vars_output]

        for uid in algeb_vars_uid:
            eq = model.get_algebraic_equations(uid)
            algeb_eqs.append(eq)

        # parse state equations
        state_vars_output = model_dict["state_var_output"]
        state_vars_uid = [var["uid"] for var in state_vars_output]

        for uid in state_vars_uid:
            eq = model.get_state_equations(uid)
            state_eqs.append(eq)



    for k, elm in enumerate(grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)):

        for e, elm in enumerate(grid.get_injection_devices_iter()):
            # rms model
            model = elm.rms_model.model

            # create dict

            model_dict = model.to_dict()

            # parse algebraic equations
            algeb_vars_output = model_dict["algebraic_var_output"]
            algeb_vars_uid = [var["uid"] for var in algeb_vars_output]

            for uid in algeb_vars_uid:
                eq = model.get_algebraic_equations(uid)
                algeb_eqs.append(eq)

            # parse state equations
            state_vars_output = model_dict["state_var_output"]
            state_vars_uid = [var["uid"] for var in state_vars_output]

            for uid in state_vars_uid:
                eq = model.get_state_equations(uid)
                state_eqs.append(eq)

    return state_eqs, algeb_eqs

class EquationsWrapp:

    def __init__(self, grid: MultiCircuit):

        self.state_eqs, self.algeb_eqs = get_equations(grid=grid)
        pdb.set_trace()

