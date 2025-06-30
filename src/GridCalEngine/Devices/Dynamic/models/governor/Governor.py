# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.Devices.Dynamic.utils.laplace_tf import LaplaceTF
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class Governor(DynamicModelTemplate):
    "This class contains the variables needed for the Governor generic model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):

        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynGovernorModel)
        
        # index
        self.bus_idx = IdxDynParam(symbol='Bus', 
                                  info="idx of the bus",
                                  id=[])
        
        self.device_idx = IdxDynParam(symbol='GENCLS',
                                      info='device id per bus',
                                      id=[])
        
        # parameters
        self.Kp = NumDynParam(symbol='Kp',
                             info='gain',
                             value=[])

        
        self.omega_ref = NumDynParam(symbol='omega_ref',
                             info='synchronous generator speed reference',
                             value=[])
        
        # state
        self.omega = ExternState(name='omega',
                                 symbol='omega',
                                 src='omega',
                                 indexer=self.device_idx)
        
        # algebraic
        self.tm = AlgebVar(name='tm',
                           symbol='tm',
                           eq='- Kp * (omega - omega_ref) - tm')
        
        