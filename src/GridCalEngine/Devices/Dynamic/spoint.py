# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

class Spoint:
    def __init__(self, component_name):
        self.name = component_name

        self.f = []
        self.g = []

        self.statVars = []
        self.algebVars = []
        self.externVars = []
        self.externStates = []
        self.externAlgebs = []
        self.aliasAlgebs = []
        self.aliasStats = []


        self.numdynParam = []
        self.idxdynParam = []
        self.extdynParam = []

    def add_statvars(self, expr):
        self.statVars.append(expr.symbol)
        self.f.append(expr.eq)

    def add_algebvars(self, expr):
        self.algebVars.append(expr.symbol)
        self.g.append(expr.eq)

    def add_externvars(self, expr):
        self.externVars.append(expr.symbol)

    def add_externstates(self, expr):
        self.externStates.append(expr.symbol)
        self.f.append(expr.eq)

    def add_externalgebs(self, expr):
        self.externAlgebs.append(expr.symbol)
        self.g.append(expr.eq)

    def add_aliasalgebs(self, expr):
        self.aliasAlgebs.append(expr.symbol)
        self.g.append(expr.eq)

    def add_aliastats(self, expr):
        self.aliasStats.append(expr.symbol)
        self.f.append(expr.eq)

    def add_numdynparam(self, expr):
        self.numdynParam.append(expr.symbol)

    def add_idxdynparam(self, expr):
        self.idxdynParam.append(expr.symbol)

    def add_extparam(self, expr):
        self.extParam.append(expr.symbol)


