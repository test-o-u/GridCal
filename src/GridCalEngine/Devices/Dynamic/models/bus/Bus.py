# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class Bus(DynamicModelTemplate):
    """
    This class contains the variables needed for the Bus model
    """
    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        """
        Bus class constructor
        :param name: Name of the Bus
        :param code: secondary code
        :param idtag: UUID code
        """
    
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynBusModel)

        # network algebraic variables
        self.a = AlgebVar(name='a',
                          symbol='a',
                          init_eq='',
                          eq=''
                         )
        
        self.v = AlgebVar(name='v',
                          symbol='v',
                          init_eq='',
                          eq=''
                         )