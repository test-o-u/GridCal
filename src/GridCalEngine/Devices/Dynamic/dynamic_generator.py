# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.dynamic_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar


class DynSynchronousModel(DynamicModelTemplate):
    "This class contains the variables needed for the dynamic simulation"

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
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
        self.delta = StatVar('delta', 'delta0', 'u * (2 * pi * fn) * (omega - 1)')  # rotor angle
        self.omega = StatVar("omega", 'u', 'u * (tm - te - D * (omega - 1))')  # vector speed

        # network algebraic variables
        self.a = ExternAlgeb("a")  # Bus voltage phase angle
        self.v = ExternAlgeb("v")  # Bus voltage magnitude

        # algebraic variables
        self.Id = AlgebVar("Id")  # d-axis current
        self.Iq = AlgebVar("Iq")  # q-axis current
        self.Vd = AlgebVar("Vd")  # v-axis voltage
        self.Vq = AlgebVar("Vq")  # q-axis voltage
        self.tm = AlgebVar("tm")  # mechanical torque
        self.te = AlgebVar("te")  # electric torque
        self.vf = AlgebVar("vf")  # excitation voltage
        self.vfc = AlgebVar("vfc")  # vf range
        self.xadIfd = AlgebVar("xadIfd")  # d-axis armator excitation current
