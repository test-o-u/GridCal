# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from src.GridCalEngine.Devices.Dynamic.models.dynamic_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

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

        # parameters
        self.bus = IdxDynParam('bus', 'interface bus id')
        self.gen = IdxDynParam('static generator index', 'gen')
        self.coi = IdxDynParam('center of inertia index', 'coi')
        self.Sn = NumDynParam('Power rating', 'Sn')
        self.Vn = NumDynParam('AC voltage rating', 'Vn')
        self.fn = NumDynParam('rated frequency', 'fn')
        self.D = NumDynParam('Damping coefficient', 'D')
        self.M = NumDynParam('machine start up time (2H)', 'M')
        self.ra = NumDynParam('armature resistance', 'ra')
        self.xl = NumDynParam('leakage reactance', 'xl')
        self.xd1 = NumDynParam('d-axis transient reactance', 'xd1')
        self.kp = NumDynParam('active power feedback gain', 'kp')
        self.kw = NumDynParam('speed feedback gain', 'kw')
        self.S10 = NumDynParam('first saturation factor','S10')
        self.S12 = NumDynParam('second saturation factor', 'S12')
        self.gammap = NumDynParam('P ratio of linked static gen', 'gammap')
        self.gammaq = NumDynParam('Q ratio of linked static gen', 'gammaq')


        # state variables
        self.delta = StatVar('delta', 'delta', 'delta0', 'u * (2 * pi * fn) * (omega - 1)')  # rotor angle
        self.omega = StatVar("omega", 'omega', 'u', 'u * (tm - te - D * (omega - 1))')  # vector speed

        # network algebraic variables
        self.a = ExternAlgeb('a', 'a', '', '-u * (vd * Id + vq * Iq)')  # Bus voltage phase angle
        self.v = ExternAlgeb('v', 'v', '', '-u * (vq * Id - vd * Iq)')  # Bus voltage magnitude

        # algebraic variables
        self.Id = AlgebVar('Id', 'Id', 'u * Id0', '')  # d-axis current
        self.Iq = AlgebVar('Iq', 'Iq', 'u * Iq0', '')  # q-axis current
        self.vd = AlgebVar('vd', 'vd', 'u * vd0', 'u * v * sin(delta - a) - vd')  # v-axis voltage
        self.vq = AlgebVar('vq', 'vq', 'u * vq0', 'u * v * cos(delta - a) - vq')  # q-axis voltage
        self.tm = AlgebVar('tm', 'tm', 'tm0', 'tm0 - tm')  # mechanical torque
        self.te = AlgebVar('te', 'te', 'u * tm0', 'u * (psid * Iq - psiq * Id) - te') # electric torque
        self.vf = AlgebVar('vf', 'vf', 'u * vf0', 'u * vf0 - vf')  # excitation voltage
        self.XadIfd = AlgebVar('XadIfd', 'XadIfd', 'u * vf0', 'u * vf0 - XadIfd')  # d-axis armator excitation current
        self.Pe = AlgebVar('Pe', 'Pe', 'u * (vd0 * Id0 + vq0 * Iq0)', 'u * (vd * Id + vq * Iq) - Pe') #active power injection
        self.Qe = AlgebVar('Qe', 'Qe', 'u * (vq0 * Id0 - vd0 * Iq0)', 'u * (vq * Id - vd * Iq) - Qe') #reactive power injection
