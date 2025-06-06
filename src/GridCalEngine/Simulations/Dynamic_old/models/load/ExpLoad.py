# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union, List
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Devices.Dynamic.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Devices.Dynamic.dyn_param import NumDynParam, IdxDynParam

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

        # to fill all the params and variables above from json file
        self.idx_dyn_param: List[IdxDynParam] = list()
        self.num_dyn_param: List[NumDynParam] = list()
        self.stat_var: List[StatVar] = list()
        self.algeb_var: List[AlgebVar] = list()
        self.ext_state_var: List[ExternState] = list()
        self.ext_algeb_var: List[ExternAlgeb] = list()

        self.comp_name = list()
        self.comp_code = list()

        # connexion status
        self.u = list()

        # parameters
        self.bus = IdxDynParam(symbol='Bus',
                               info='Load bus',
                               ident=[],
                               connection_point = 'ExpLaod')

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
        self.p = ExternAlgeb(name='p',
                             symbol = 'p',
                             src='p',
                             indexer=self.bus, 
                             init_eq='', 
                             eq='Pl0 * q ** coeff_alfa')
        
        self.q = ExternAlgeb(name='q',
                             symbol = 'q',
                             src='q',
                             indexer=self.bus,
                             init_eq='',
                             eq='Ql0 * q ** coeff_beta')