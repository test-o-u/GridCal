# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class ACLine(DynamicModelTemplate):
    "This class contains the variables needed for the AC line model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
    
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynACLineModel)

        # parameters 
        # TODO: 
        # - ANDES separates ModeData from Model for efficieny reasons (I guess). What do we want to do? 
        # - is there a reason why it is called symbol and not model as in ANDES?
        self.bus1 = IdxDynParam(symbol='Bus', 
                                info="idx of from bus")
        
        self.bus2 = IdxDynParam(symbol='Bus',
                                info="idx of to bus")

        self.g = NumDynParam(symbol='g',
                             info='shared shunt conductance',
                             value=0.0)
        
        self.b = NumDynParam(symbol='b',
                             info='shared shunt susceptance',
                             value=0.0)
        
        self.bsh = NumDynParam(symbol='bsh',
                               info='from/to-side shunt susceptance',
                               value=0.0)

        # network algebraic variables 
        # TODO: 
        # - discuss modeling. Here a pi-model is considered and the power flow equations are derived according to it, while in ANDES they apply some transformations first.
        # - check if naming makes sense.
        self.a1 = ExternAlgeb(index = 0,
                              name='',
                              symbol='a1',
                              init_eq='', 
                              eq='-u * (v1 ** 2 * g  - \
                                    v1 * v2 * (g * cos(a1 - a2) + \
                                        b * sin(a1 - a2)))')  
        
        self.v1 = ExternAlgeb(index = 1,
                              name='v1',
                              symbol='v1',
                              init_eq='', 
                              eq='-u * (- v1 ** 2 * (b + bsh / 2) - \
                                    v1 * v2 * (g * sin(a1 - a2) - \
                                        b * cos(a1 - a2)))')
         
        self.a2 = ExternAlgeb(index = 2,
                              name='a2',
                              symbol='a2',
                              init_eq='', 
                              eq='u * (v2 ** 2 * g21  - \
                                    v2 * v1 * (g21 * cos(a2 - a1) + \
                                        b21 * sin(a2 - a1)))')  
        
        self.v2 = ExternAlgeb(index = 3,
                              name='v2',
                              symbol='v2',
                              init_eq='', 
                              eq='u * (- v2 ** 2 * (b21 + bsh / 2) - \
                                    v2 * v1 * (g21 * sin(a2 - a1) - \
                                        b21 * cos(a2 - a1)))')  