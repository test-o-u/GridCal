# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class GENCLS(DynamicModelTemplate):
    "This class contains the parameters and variables needed for the GENCLS model"
    # TODO: check GENCLS model.

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        """
        GENCLS class constructor
        :param name: Name of the GENCLS
        :param code: secondary code
        :param idtag: UUID code
        """
        
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

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
                               id=[],
                               connection_point = 'GENCLS')
        
        self.fn = NumDynParam(symbol='fn',
                              info='rated frequency',
                              value=[])
        
        self.D  = NumDynParam(symbol='D',
                              info='damping coefficient',
                              value=[])
        
        self.M  = NumDynParam(symbol='M',
                              info='machine start up time (2H)',
                              value=[])

        self.ra = NumDynParam(symbol='ra', 
                              info='armature resistance',
                              value=[])
        
        self.xd = NumDynParam(symbol='xd',
                              info='d-axis transient reactance',
                              value=[])
        
        self.tm = NumDynParam(symbol='tm',
                              info='uncontrolled mechanical torque',
                              value=[])
        
        self.vf = NumDynParam(symbol='vf',
                              info='uncontrolled exitation voltage',
                              value=[]) 

        # state variables
        self.delta = StatVar(name='delta', 
                             symbol='delta', 
                             init_eq='delta0', 
                             eq='(2 * pi * fn) * (omega - 1)')   
                              
        self.omega = StatVar(name='omega', 
                             symbol='omega', 
                             init_eq='omega_0', 
                             eq='(-tm / M + t_e / M - D / M * (omega - 1))')

        # algebraic variables
        self.psid = AlgebVar(name='psid',
                             symbol='psid',
                             init_eq='psid0',
                            #  eq='(ra * i_q + vq) - psid')
                             eq='(-ra * i_q + v_q) - psid')
        
        self.psiq = AlgebVar(name='psiq',
                             symbol='psiq',
                             init_eq='psiq0',
                            #  eq='(ra * i_d + v_d) - psiq')
                             eq='(-ra * i_d + v_d) - psiq')
        
        self.i_d = AlgebVar(name='i_d', 
                           symbol='i_d', 
                           init_eq='i_d0', 
                           eq='psid + xd * i_d - vf') # vd
                                                         
        self.i_q = AlgebVar(name='i_q', 
                           symbol='i_q', 
                           init_eq='i_q0', 
                           eq='psiq + xd * i_q') # vd
                                                     
        self.v_d = AlgebVar(name='v_d',
                           symbol='v_d',
                           init_eq='v_d0',
                           eq='q * sin(delta - p) - v_d')
                                    
        self.v_q = AlgebVar(name='v_q',
                           symbol='v_q',
                           init_eq='v_q0',
                           eq='q * cos(delta - p) - v_q')
                                 
        self.t_e = AlgebVar(name='t_e',
                           symbol='t_e',
                           init_eq='tm',
                           eq='(psid * i_q - psiq * i_d) - t_e')
                     
        self.P_e = AlgebVar(name='P_e',
                           symbol='P_e',
                           init_eq='(v_d0 * i_d0 + v_q0 * i_q0)',
                           eq='(v_d * i_d + v_q * i_q) - P_e')
                                
        self.Q_e = AlgebVar(name='Q_e',
                           symbol='Q_e',
                           init_eq='(v_q0 * i_d0 - v_d0 * i_q0)',
                           eq='(v_q * i_d - v_d * i_q) - Q_e')

        self.p = ExternAlgeb(name='p',
                             symbol='p',
                             src='p',
                             indexer=self.bus,
                             init_eq='',
                             eq='(v_d * i_d + v_q * i_q)')

        self.q = ExternAlgeb(name='q',
                             symbol='q',
                             src='q',
                             indexer=self.bus,
                             init_eq='',
                             eq='(v_q * i_d - v_d * i_q)')


        # network algebraic variables 
        # TODO: 
        # -check naming
        # -check how they are exported
