# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam, ExtDynParam

class Slack(DynamicModelTemplate):
    "This class contains the parameters and variables needed for the Slack model"

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        """
        Slack class constructor
        :param name: Name of the Slack
        :param code: secondary code
        :param idtag: UUID code
        """
        
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSlackModel)

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
                               info='interface bus id',
                               ident=[],
                               connection_point = 'Slack')
        
        self.Sn = NumDynParam(symbol='Sn',
                              info='power rating',
                              value=[])
        
        self.Vn  = NumDynParam(symbol='Vn',
                              info='AC voltage rating',
                              value=[])
        
        self.P_e0  = NumDynParam(symbol='P_e0',
                              info='active power set point in system base',
                              value=[])

        self.Q_e0 = NumDynParam(symbol='Q_e0',
                              info='reactive power set point in system base',
                              value=[])
        
        self.pmax = NumDynParam(symbol='pmax',
                              info='maximum active power in system base',
                              value=[])
        
        self.pmin = NumDynParam(symbol='pmin',
                              info='minimum active power in system base',
                              value=[])
        
        self.qmax = NumDynParam(symbol='qmax',
                              info='maximum reactive power in system base',
                              value=[])

        self.qmin = NumDynParam(symbol='qmin',
                                info='minimum reactive power in system base',
                                value=[])

        self.q0 = NumDynParam(symbol='q0',
                                info='voltage set point',
                                value=[])

        self.vmax = NumDynParam(symbol='vmax',
                                info='maximum voltage voltage',
                                value=[])

        self.vmin = NumDynParam(symbol='vmin',
                                info='minimum allowed voltage',
                                value=[])

        self.ra = NumDynParam(symbol='ra',
                                info='armature resistance',
                                value=[])

        self.xs = NumDynParam(symbol='xs',
                                info='armature reactance',
                                value=[])

        self.p0 = NumDynParam(symbol='p0',
                                info='reference angle set point',
                                value=[])

        self.busv0 = ExtDynParam(symbol='bus_v0',
                                info='',
                                value=[])

        self.busa0 = ExtDynParam(symbol='bus_a0',
                                info='',
                                value=[])


        # algebraic variables
                     
        self.P_e_slack = AlgebVar(name='P_e_slack',
                           symbol='P_e_slack',
                           init_eq='p0',
                           eq= 'p0-p + pmin-P_e_slack + pmax-P_e_slack')
                                
        self.Q_e_slack = AlgebVar(name='Q_e_slack',
                           symbol='Q_e_slack',
                           init_eq='q0',
                           eq= 'q0-q + qmin-Q_e_slack + qmax-Q_e_slack')

        self.p = ExternAlgeb(name='p',
                             symbol='p',
                             src='p',
                             indexer=self.bus,
                             init_eq='a0+busa0',
                             eq='(-p)')

        self.q = ExternAlgeb(name='q',
                             symbol='q',
                             src='q',
                             indexer=self.bus,
                             init_eq='v0+busv0',
                             eq='(-q)')