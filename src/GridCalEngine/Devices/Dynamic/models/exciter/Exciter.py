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

class Exciter(DynamicModelTemplate):
    "This class contains the variables needed for the Exciter generic model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.Exciter)
        
        # index
        self.bus_idx = IdxDynParam(symbol='Bus', 
                                  info="idx of from the synchronous generator",
                                  id=[])
        
        # parameters
        self.G = NumDynParam(symbol='G',
                             info='gain',
                             value=[])

        self.T = NumDynParam(symbol='T',
                             info='first order field voltage dynamic time constant',
                             value=[])
        
        self.v_ref = NumDynParam(symbol='v_ref',
                             info='bus(terminal) voltage reference',
                             value=[])
        
        # variables 
        self.v = ExternAlgeb(name='v',
                             symbol='v',
                             src='v',
                             indexer=self.bus_idx,
                             init_eq='',
                             eq='')
        
        self.vf = StatVar(name='vf',
                          symbol='vf',
                          t_const=self.T,
                          tf=LaplaceTF("G / (T*s + 1)", 
                                        input='v - v_ref', 
                                        output='vf'))
                            
        
