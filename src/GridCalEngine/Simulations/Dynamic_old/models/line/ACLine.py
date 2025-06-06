# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union, List
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Devices.Dynamic.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Devices.Dynamic.dyn_param import NumDynParam, IdxDynParam

class ACLine(DynamicModelTemplate):
    "This class contains the parameters and variables needed for the AC line model."

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        """
        ACLine class constructor
        :param name: Name of the ACLine
        :param code: secondary code
        :param idtag: UUID code
        """

        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynACLineModel)

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
        self.u =list()

        # parameters 
        # TODO: 
        # - ANDES separates ModeData from Model for efficieny reasons (I guess). What do we want to do?
        self.bus1 = IdxDynParam(symbol='Bus',
                                info="idx of from bus",
                                ident=[],
                                connection_point = 'ACLine_origin',
                                name="")
        
        self.bus2 = IdxDynParam(symbol='Bus',
                                info="idx of to bus",
                                ident=[],
                                connection_point='ACLine_end',
                                name="")

        self.g = NumDynParam(symbol='g',
                             info='shared shunt conductance',
                             value=[])
        
        self.b = NumDynParam(symbol='b',
                             info='shared shunt susceptance',
                             value=[])
        
        self.bsh = NumDynParam(symbol='bsh',
                               info='from/to-side shunt susceptance',
                               value=[])

        # network algebraic variables 
        # TODO: 
        # - discuss modeling. Here a pi-model is considered and the power flow equations are derived according to it, while in ANDES they apply some transformations first.
        # - check if naming makes sense.
        self.P_origin = ExternAlgeb(name='P_origin',
                              symbol = 'P_origin',
                              src='p',
                              indexer=self.bus1, 
                              init_eq='', 
                              eq='(Q_origin ** 2 * g  - \
                                    Q_origin * Q_end * (g * cos(P_origin - P_end) + \
                                        b * sin(P_origin - P_end)))')

        self.Q_origin = ExternAlgeb(name='Q_origin',
                              symbol='Q_origin',
                              src='q',
                              indexer=self.bus1,
                              init_eq='',
                              eq='(- Q_origin ** 2 * (b + bsh / 2) - \
                                            Q_origin * Q_end * (g * sin(P_origin - P_end) - \
                                                b * cos(P_origin - P_end)))')
        
        self.P_end = ExternAlgeb(name='P_end',
                              symbol = 'P_end',
                              src='p',
                              indexer=self.bus2, 
                              init_eq='', 
                              eq='(Q_end ** 2 * g  - \
                                    Q_end * Q_origin * (g * cos(P_end - P_origin) + \
                                        b * sin(P_end - P_origin)))')
        
        self.Q_end = ExternAlgeb(name='Q_end',
                              symbol = 'Q_end',
                              src='q',
                              indexer=self.bus2, 
                              init_eq='', 
                              eq='(- Q_end ** 2 * (b + bsh / 2) - \
                                    Q_end * Q_origin * (g * sin(P_end - P_origin) - \
                                        b * cos(P_end - P_origin)))')