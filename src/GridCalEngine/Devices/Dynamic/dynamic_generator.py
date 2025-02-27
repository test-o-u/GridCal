# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.dynamic_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import DynVar


class DynSynchronousModel(DynamicModelTemplate):
    "This class contains the variables and equations needed for the dynamic simulation"

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None],
                 system_eq):
        """

        :param system_eq:
        :param delta:
        :param omega:
        :param a:
        :param v:
        :param Id:
        :param Iq:
        :param Vd:
        :param Vq:
        :param tm:
        :param te:
        :param vf:
        :param vfc:
        :param xadIfd:
        """
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

        # state variables
        self.delta = DynVar("delta")  # rotor angle
        self.omega = DynVar("omega")  # vector speed

        # network algebraic variables
        self.a = DynVar("a")  # Bus voltage phase angle
        self.v = DynVar("v")  # Bus voltage magnitude

        # algebraic variables
        self.Id = DynVar("Id")  # d-axis current
        self.Iq = DynVar("Iq")  # q-axis current
        self.Vd = DynVar("Vd")  # v-axis voltage
        self.Vq = DynVar("Vq")  # q-axis voltage
        self.tm = DynVar("tm")  # mechanical torque
        self.te = DynVar("te")  # electric torque
        self.vf = DynVar("vf")  # excitation voltage
        self.vfc = DynVar("vfc")  # vf range
        self.xadIfd = DynVar("xadIfd")  # d-axis armator excitation current

        # system equations
        self.equations = system_eq
