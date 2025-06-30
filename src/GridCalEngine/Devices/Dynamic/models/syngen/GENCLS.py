# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam

class GENCLS(DynamicModelTemplate):
    "This class contains the variables needed for the GENCLS model"
    # TODO: check GENCLS model.

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        
        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

        # indexes
        self.bus_idx = IdxDynParam(symbol='Bus', 
                               info='interface bus id',
                               id=[])
        
        # self.exciter_idx = IdxDynParam(symbol='Exciter', 
        #                        info='exciter id per bus',
        #                        id=[])
        
        self.governor_idx = IdxDynParam(symbol='Governor',
                               info='governor id per bus',
                               id=[])
        
        # parameters    
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
        
        self.xq = NumDynParam(symbol='xq',
                              info='q-axis transient reactance',
                              value=[])
        
        # self.tm = NumDynParam(symbol='tm',
        #                       info='uncontrolled mechanical torque',
        #                       value=[])
        
        self.vf = NumDynParam(symbol='vf',
                              info='uncontrolled exitation voltage',
                              value=[]) 

        # state variables
        self.delta = StatVar(name='delta', 
                             symbol='delta', 
                             eq='(2 * pi * fn) * (omega - 1)')   
                              
        self.omega = StatVar(name='omega', 
                             symbol='omega', 
                             eq='(tm - te  - D * (omega - 1))',
                             t_const=self.M)
        
        # self.vf = ExternState(name='vf',
        #                       symbol='vf',
        #                       src='vf',
        #                       indexer=self.exciter_idx)

        # algebraic variables
        self.psid = AlgebVar(name='psid',
                             symbol='psid',
                             eq='(ra * i_q + vq) - psid') 
        
        self.psiq = AlgebVar(name='psiq',
                             symbol='psiq',
                             eq='(ra * i_d + vd) + psiq') # Note: sign needs to be discussed.
        
        self.i_d = AlgebVar(name='i_d', 
                           symbol='i_d', 
                           eq='psid + xq * i_d - vf')
                                                         
        self.i_q = AlgebVar(name='i_q', 
                           symbol='i_q',
                           eq='psiq + xq * i_q')
                                                     
        self.vd = AlgebVar(name='vd', 
                           symbol='vd',
                           eq='v * sin(delta - a) - vd')  
                                    
        self.vq = AlgebVar(name='vq', 
                           symbol='vq',
                           eq='v * cos(delta - a) - vq')   
                                 
        self.te = AlgebVar(name='te', 
                           symbol='te',
                           eq='(psid * i_q - psiq * i_d) - te')   
                     
        self.Pe = AlgebVar(name='Pe',
                           symbol='Pe',
                           eq='(vd * i_d + vq * i_q) - Pe')       
                                
        self.Qe = AlgebVar(name='Qe', 
                           symbol='Qe',
                           eq='(vq * i_d - vd * i_q) - Qe')

        self.a = ExternAlgeb(name='a',
                             symbol='a',
                             src='a',
                             indexer=self.bus_idx,
                             eq='(vd * i_d + vq * i_q)')

        self.v = ExternAlgeb(name='v',
                             symbol='v',
                             src='v',
                             indexer=self.bus_idx,
                             eq='(vq * i_d - vd * i_q)')
        
        self.tm = ExternAlgeb(name='tm',
                             symbol='tm',
                             src='tm',
                             indexer=self.governor_idx)
