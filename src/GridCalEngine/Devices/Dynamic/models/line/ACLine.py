# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class ACLine(DynamicModelTemplate):
    "This class contains the variables needed for the AC line model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
    
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

        # parameters TODO: ANDES separates ModeData from Model for efficieny reasons (I guess). What do we want to do? 
        self.bus1 = IdxDynParam(symbol='Bus', #TODO: is there a reason why it is called symbol and not model as in ANDES?
                                info="idx of from bus")
        
        self.bus2 = IdxDynParam(symbol='Bus',
                                id = self.params['idx'],
                                info="idx of to bus")

        self.g = NumDynParam(symbol='g',
                               info='shared shunt conductance')
        
        self.b = NumDynParam(symbol='b',
                               info='shared shunt susceptance')
        
        self.bsh = NumDynParam(symbol='bsh',
                               info='from/to-side shunt susceptance')

        # network algebraic variables 
        # TODO: 
        # - discuss modeling. Here a pi-model is considered and the power flow equations are derived according to it, while in ANDES they apply some transformations first.
        # - check if naming makes sense.
        self.a1 = ExternAlgeb(name='a',       
                              symbol='a', 
                              init_eq='', 
                              eq='-u * (v1 ** 2 * g  - \
                                    v1 * v2 * (g * cos(a1 - a2) + \
                                        b * sin(a1 - a2)))')  
        
        self.v1 = ExternAlgeb(name='a', 
                              symbol='a', 
                              init_eq='', 
                              eq='-u * (- v1 ** 2 * (b + bsh / 2) - \
                                    v1 * v2 * (g * sin(a1 - a2) - \
                                        b * cos(a1 - a2)))')
         
        self.a2 = ExternAlgeb(name='a', 
                              symbol='a', 
                              init_eq='', 
                              eq='u * (v2 ** 2 * g21  - \
                                    v2 * v1 * (g21 * cos(a2 - a1) + \
                                        b21 * sin(a2 - a1)))')  
        
        self.v2 = ExternAlgeb(name='a', 
                              symbol='a', 
                              init_eq='', 
                              eq='u * (- v2 ** 2 * (b21 + bsh / 2) - \
                                    v2 * v1 * (g21 * sin(a2 - a1) - \
                                        b21 * cos(a2 - a1)))')  