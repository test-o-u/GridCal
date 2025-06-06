# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, DynVar
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

        # to fill all the params and variables above from json file
        self.idx_dyn_param: List[IdxDynParam] = list()
        self.num_dyn_param: List[NumDynParam] = list()
        self.stat_var: List[StatVar] = list()
        self.algeb_var: List[AlgebVar] = list()
        self.ext_state_var: List[ExternState] = list()
        self.ext_algeb_var: List[ExternAlgeb] = list()

        self.comp_name = list()
        self.comp_code = list()

        # parameters
        self.p0 = NumDynParam(symbol='p0',
                              info='initial voltage phase angle',
                              value=[])
        self.q0 = NumDynParam(symbol='q0',
                              info='initial voltage magnitude',
                              value=[])

        # network algebraic variables
        self.p = AlgebVar(name='p',
                          symbol='p',
                          init_eq='',
                          eq=''
                         )
        
        self.q = AlgebVar(name='q',
                          symbol='q',
                          init_eq='',
                          eq=''
                         )