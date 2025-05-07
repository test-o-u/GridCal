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
        """
        ExpLoad class constructor
        :param name: Name of the ExpLoad
        :param code: secondary code
        :param idtag: UUID code
        """
    
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynExpLoadModel)

        # parameters
        self.bus = IdxDynParam(symbol='Bus', 
                                info='Load bus',
                                id=[])

        self.coeff_alfa = NumDynParam(symbol='coeff_alfa',
                                info='Active power load exponential coefficient.',
                                value=[])
        
        self.coeff_beta = NumDynParam(symbol='coeff_beta',
                                info='Active power load exponential coefficient.',
                                value=[])
        
        # self.beta = NumDynParam(symbol='beta',
        #                         info='Reactive Power load exponential coefficient.',
        #                         value=[])
        
        self.Pl0 = NumDynParam(symbol='Pl0',
                                info='Active Power load base.',
                                value=[])
        
        self.Ql0 = NumDynParam(symbol='Ql0',
                                info='Reactive Power load base.',
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
                             eq='Pl0 * v ** coeff_alfa')  
        
        self.v = ExternAlgeb(name='v',
                             symbol = 'v',
                             src='v',
                             indexer=self.bus,
                             init_eq='',
                             eq='Ql0 * v ** coeff_beta')