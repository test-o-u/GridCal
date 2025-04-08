# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, AliasState, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class GENCLS(DynamicModelTemplate):
    "This class contains the variables needed for the GENCLS model"
    # TODO: check GENCLS model.

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

        # parameters
        self.bus = IdxDynParam(symbol='Bus', 
                               info='interface bus id',
                               id=[])
        
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
                             eq='(tm - te - D * (omega - 1))')                          

        # algebraic variables
        self.psid = AlgebVar(name='psid',
                             symbol='psid',
                             init_eq='psid0',
                             eq='(ra * i_q + vq) - psid')
        
        self.psiq = AlgebVar(name='psiq',
                             symbol='psiq',
                             init_eq='psiq0',
                             eq='(ra * i_d + vd) - psiq')
        
        self.i_d = AlgebVar(name='i_d', 
                           symbol='i_d', 
                           init_eq='i_d0', 
                           eq='psid + vd * i_d - vf')
                                                         
        self.i_q = AlgebVar(name='i_q', 
                           symbol='i_q', 
                           init_eq='i_q0', 
                           eq='psiq + vd * i_q')
                                                     
        self.vd = AlgebVar(name='vd', 
                           symbol='vd', 
                           init_eq='vd0', 
                           eq='v * sin(delta - a) - vd')  
                                    
        self.vq = AlgebVar(name='vq', 
                           symbol='vq', 
                           init_eq='vq0', 
                           eq='v * cos(delta - a) - vq')   
                                 
        self.te = AlgebVar(name='te', 
                           symbol='te', 
                           init_eq='tm', 
                           eq='(psid * i_q - psiq * i_d) - te')   
                     
        self.Pe = AlgebVar(name='Pe',
                           symbol='Pe', 
                           init_eq='(vd0 * i_d0 + vq0 * i_q0)', 
                           eq='(vd * i_d + vq * i_q) - Pe')       
                                
        self.Qe = AlgebVar(name='Qe', 
                           symbol='Qe', 
                           init_eq='(vq0 * i_d0 - vd0 * i_q0)', 
                           eq='(vq * i_d - vd * i_q) - Qe')

        self.a = ExternAlgeb(name='a',
                             symbol='a',
                             src='a',
                             indexer=self.bus,
                             init_eq='',
                             eq='(vd * i_d + vq * i_q)')

        self.v = ExternAlgeb(name='v',
                             symbol='v',
                             src='v',
                             indexer=self.bus,
                             init_eq='',
                             eq='(vq * i_d - vd * i_q)')


        # network algebraic variables 
        # TODO: 
        # -check naming
        # -check how they are exported
