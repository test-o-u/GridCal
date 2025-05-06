# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

'Class to store variables'

class DynVar:
    def __init__(self, name: str, symbol: str, eq: str,  init_eq: str):
        self.name = name
        self.symbol = symbol
        self.init_eq = init_eq
        self.eq = eq

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AlgebVar(DynVar):
    def __init__(self, name: str, symbol: str, eq: str, init_eq=None):
        DynVar.__init__(self,
                        name=name,
                        symbol=symbol,
                        init_eq=init_eq,
                        eq=eq)
        self.var_type = 'y'


class StatVar(DynVar):
    def __init__(self, name: str, symbol: str, eq=None, init_eq=None, t_const=None, tf=None):
        DynVar.__init__(self,
                        name=name,
                        symbol=symbol,
                        init_eq=init_eq,
                        eq=eq
                        )
        self.var_type = 'x'
        self.t_const = t_const
        self.tf = tf
    
        if tf is not None:
            self._convert_tf(tf)

    def _convert_tf(self, tf):
        rhs_expr = tf.process_tf()

        self.eq = str(rhs_expr)
        # self.t_const = lhs_coeff

class ExternVar(DynVar):
    def __init__(self, name: str, symbol: str, src: str, indexer, eq=None, init_eq=None):
        DynVar.__init__(self, name=name,
                        symbol=symbol,
                        eq=eq,
                        init_eq=init_eq)
        self.src = src
        self.indexer = indexer 

class ExternState(ExternVar):
    def __init__(self, name: str, symbol: str, src: str, indexer, eq=None, init_eq=None):
        ExternVar.__init__(self, name=name,
                           symbol=symbol,
                           src=src,
                           indexer=indexer,
                           eq=eq,
                           init_eq=init_eq)
        self.var_type = 'x'

class ExternAlgeb(ExternVar):
    def __init__(self, name: str, symbol: str, src: str, indexer, eq=None, init_eq=None):
        ExternVar.__init__(self, name=name,
                           symbol=symbol,
                           src=src,
                           indexer=indexer,
                           eq=eq,
                           init_eq=init_eq)
        self.var_type = 'y'

