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
        self.bus = IdxDynParam(symbol='bus', 
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
                             eq='u * (2 * pi * fn) * (omega - 1)')                         
        self.omega = StatVar(name='omega', 
                             symbol='omega', 
                             init_eq='omega_0', 
                             eq='u * (tm - te - D * (omega - 1))')                          

        # algebraic variables
        self.psid = AlgebVar(name='d-axis flux',
                             symbol=r'\psi_d',
                             init_eq='psid0',
                             eq='u * (ra*Iq + vq) - psid')
        self.psiq = AlgebVar(name='q-axis flux',
                             symbol=r'\psi_d',
                             init_eq='psiq0',
                             eq='u * (ra*Id + vd) - psid')
        self.Id = AlgebVar(name='Id', 
                           symbol='Id', 
                           init_eq='Id0', 
                           eq='psid + xq * Id - vf')                                                     
        self.Iq = AlgebVar(name='Iq', 
                           symbol='Iq', 
                           init_eq='Iq0', 
                           eq='psiq + xq * Iq')                                                    
        self.vd = AlgebVar(name='vd', 
                           symbol='vd', 
                           init_eq='vd0', 
                           eq='u * v * sin(delta - a) - vd')                              
        self.vq = AlgebVar(name='vq', 
                           symbol='vq', 
                           init_eq='vq0', 
                           eq='u * v * cos(delta - a) - vq')                            
        self.te = AlgebVar(name='te', 
                           symbol='te', 
                           init_eq='tm', 
                           eq='u * (psid * Iq - psiq * Id) - te')                
        self.Pe = AlgebVar(name='Pe', 
                           symbol='Pe', 
                           init_eq='u * (vd0 * Id0 + vq0 * Iq0)', 
                           eq='u * (vd * Id + vq * Iq) - Pe')                               
        self.Qe = AlgebVar(name='Qe', 
                           symbol='Qe', 
                           init_eq='u * (vq0 * Id0 - vd0 * Iq0)', 
                           eq='u * (vq * Id - vd * Iq) - Qe')                            

        # network algebraic variables 
        # TODO: 
        # -check naming
        # -check how they are exported
        self.a = ExternAlgeb(name='a', 
                             symbol = 'a',
                             src='a',
                             indexer=self.bus, 
                             init_eq='', 
                             eq='u * (vd * Id + vq * Iq)')                                 
        self.v = ExternAlgeb(name='v', 
                             symbol = 'v',
                             src='v', 
                             indexer=self.bus,
                             init_eq='', 
                             eq='u * (vq * Id - vd * Iq)')                                
