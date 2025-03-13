# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class Bus(DynamicModelTemplate):
    "This class contains the variables needed for the Bus model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
    
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynBusModel)

        # network algebraic variables 
        # TODO: check the naming and see if it really make sense to keep them like this.
        self.a = AlgebVar(name='a',
                          symbol='a',
                          init_eq='',
                          eq=''
                         )
        
        self.v = AlgebVar(name='a',
                          symbol='a',
                          init_eq='',
                          eq=''
                         )