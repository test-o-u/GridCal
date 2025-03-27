# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class ExpLoad(DynamicModelTemplate):
    "This class contains the variables needed for the Exponential Load model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
    
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynExpLoadModel)

        # parameters
        self.bus = IdxDynParam(symbol='Bus', 
                                info='Load bus',
                                id=[])

        self.alfa = NumDynParam(symbol='alfa',
                                info='Active power load exponential coefficient.',
                                value=[])
        
        self.beta = NumDynParam(symbol='beta',
                                info='Reactive Power load exponential coefficient.',
                                value=[])

        # network algebraic variables 
        # TODO:
        # - check if naming make sense 
        # - indexing is missing 
        self.a = ExternAlgeb(name='a',
                             symbol = 'a', 
                             src='a',
                             indexer=self.bus, 
                             init_eq='', 
                             eq='u * Pl0 * v ** alfa')  
        
        self.v = ExternAlgeb(name='v',
                             symbol = 'v',  
                             src='v',
                             indexer=self.bus,  
                             init_eq='', 
                             eq='u * Ql0 * v ** beta')