# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.  
# SPDX-License-Identifier: MPL-2.0

# References:
# Cui et al., Hybrid Symbolic-Numeric Framework for Power 
# System Modeling and Analysis, https://arxiv.org/pdf/2002.09455
# Lara et al., Revisiting Power Systems Time-domain Simulation 
# Methods and Models, https://arxiv.org/pdf/2301.10043


import numpy as np


# Generic definitions
class DynamicModel:
    def __init__(self):
        # self._model = None
        # self._model_name = None
        # self._model_type = None
        # self._model_data = None
        # self._model_parameters = None
        # self._model_states = None
        # self._model_inputs = None
        # self._model_outputs = None
        # self._model_states = None
        # self._model_state_matrix = None
        # self._model_output
        
        pass

class Block:
    """
    Base class for all blocks in the system
    """
    def __init__(self,
                 name: str):
        self.name = name
        self.vars = dict()
        self.jac_trip = JacTrip()


class JacTrip:
    def __init__(self):
        """
        Build a structure to store the triplets (row, col, value) of the jacobian matrix
        """
        self.i = dict(list)
        self.j = dict(list)
        self.v = dict(list)

    def add_entries(self, jac_id, i, j, v):
        """
        Add a triplet to a particular subjacobian matrix

        :param jac_id: 
        :type jac_id: uuid idtag for the jacobian matrix
        :param i: row identifier
        :type i: int
        :param j: col identifier
        :type j: int
        :param v: entry value
        :type v: float
        """
        self.i[jac_id].append(i)
        self.j[jac_id].append(j)
        self.v[jac_id].append(v)
    
    def get_my_jac(self, jac_id):
        return self.i[jac_id], self.j[jac_id], self.v[jac_id]

        
class DynamicParentData:
    def __init__(self):
        self.params = dict()
        self.num_params = dict()
        self.parent_id = dict()

 
class BasicVar:
    def __init__(self,
                 name,
                 var_str,
                 eq_str):
        self.name = name
        self.var_str = var_str
        self.eq_str = eq_str
        
        # placeholder, will eventually have length of number of time steps
        self.v: np.ndarray = np.array([], dtype=float)
        self.e: np.ndarray = np.array([], dtype=float)


class AlgebraicEq(BasicVar):
    """
    To be used when defining algebraic equations
    """
    eq_str = 'g'  # we are looking for g(x) = 0
    var_str = 'y'  # name of the variable, i.e., of the function?


class DifferentialEq(BasicVar):
    """
    To be used when defining differential equations
    """
    eq_str = 'f'
    var_str = 'x'

    def __init__(self,
                 t_const: float):
        self.t_const = t_const

        
class ExtVar(BasicVar):
    def __init__(self):
        super().__init__()

        self.model = None
        self.parent = None


class BaseParam:
    def __init__(self,
                 name:str = None):
        self.name = name


class IdxParam(BaseParam):
    def __init__(self,
                 model:str = None):
        """

        :param model: should model be a uuid pointer? Or just a regular string?
        """
        super().__init__()
        self.model = model


class NumParam(BaseParam):
    def __init__(self,
                 value:float = None):
        super().__init__()
        self.value = value


class ExtParam(NumParam):
    def __init__(self,
                 model:str = None):

        super().__init__()
        self.model = model
        

class ExtState(ExtVar):
    e_code = 'f'
    r_code = 'h'
    v_code = 'x'


class ExtAlgeb(ExtVar):
    e_code = 'a'
    r_code = 'b'
    v_code = 'c'


class Algeb(BasicVar):
    e_code = 'a'
    v_code = 'y'


class Const():
    def __init__(self, 
                 v_str:str = '',
                 val_num:str = ''):
        self.v_str = v_str
        self.val_num = val_num


class Model():
    def __init__(self):
        self.state = dict()
        self.state_extern = dict()
        self.algeb = dict()
        self.algeb_extern = dict()
        self.params = dict()

        self.rhs_f = dict()
        self.rhs_g = dict()


# Actual implementation of a dynamic model
class TurbineGenData(DynamicParentData):
    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen')
        self.Tn = NumParam(value=0.5)
        self.wref0 = NumParam(value=1.0)

        
class TurbineGenBase(Model):
    def __init__(self):
        Model.__init__(self)

        self.Snom = ExtParam(name='Snom', 
                             model='SynGen')
        self.Ug = ExtParam(name='Ug',
                           model='SynGen')

        self.omega = ExtState(name='omega',
                              model='SynGen')

        self.tm = ExtAlgeb(name='tm',
                           model='SynGen',
                           e_str='ue * (pout - tm0)')

        self.paux = Algeb(v_str='paux0',
                          e_str='paux0 - paux')

        self.ue = Const(v_str='Ug')

        self.paux0 = Const(v_str='0.0')

        self.pout = Algeb(v_str='ue * tm0')

        self.wref = Algeb(v_str='wref0',
                          e_str='wref0 - wref')
    

class TurbineGenImp(TurbineGenData, TurbineGenBase):
    def __init__(self):
        TurbineGenData.__init__(self)
        TurbineGenBase.__init__(self)
