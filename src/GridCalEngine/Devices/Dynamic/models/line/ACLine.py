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

        # parameters 
        # TODO: 
        # - ANDES separates ModeData from Model for efficieny reasons (I guess). What do we want to do?
        self.bus1 = IdxDynParam(symbol='Bus', 
                                info="idx of from bus",
                                id=[])
        
        self.bus2 = IdxDynParam(symbol='Bus',
                                info="idx of to bus",
                                id=[])

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
        self.a_origin = ExternAlgeb(name='a_origin',
                              symbol = 'a_origin',
                              src='a',
                              indexer=self.bus1, 
                              init_eq='', 
                              eq='(v_origin ** 2 * g  - \
                                    v_origin * v_end * (g * cos(a_origin - a_end) + \
                                        b * sin(a_origin - a_end)))')

        self.v_origin = ExternAlgeb(name='v_origin',
                              symbol='v_origin',
                              src='v',
                              indexer=self.bus1,
                              init_eq='',
                              eq='(- v_origin ** 2 * (b + bsh / 2) - \
                                            v_origin * v_end * (g * sin(a_origin - a_end) - \
                                                b * cos(a_origin - a_end)))')
        
        self.a_end = ExternAlgeb(name='a_end',
                              symbol = 'a_end',
                              src='a',
                              indexer=self.bus2, 
                              init_eq='', 
                              eq='(v_end ** 2 * g  - \
                                    v_end * v_origin * (g * cos(a_end - a_origin) + \
                                        b * sin(a_end - a_origin)))')
        
        self.v_end = ExternAlgeb(name='v_end',
                              symbol = 'v_end',
                              src='v',
                              indexer=self.bus2, 
                              init_eq='', 
                              eq='(- v_end ** 2 * (b + bsh / 2) - \
                                    v_end * v_origin * (g * sin(a_end - a_origin) - \
                                        b * cos(a_end - a_origin)))')