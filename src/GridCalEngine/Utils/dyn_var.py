# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

'Class to store variables info'


class DynVar:
    def __init__(self, name: str, init_eq: str, eq: str):
        self.name = name
        self.init_eq = init_eq
        self.eq = eq

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AlgebVar(DynVar):
    def __init__(self, name: str, init_eq: str, eq: str):
        DynVar.__init__(self, name=name,
                        init_eq=init_eq,
                        eq=eq)


class StatVar(DynVar):
    def __init__(self, name: str, init_eq: str, eq: str):
        DynVar.__init__(self, name=name,
                        init_eq=init_eq,
                        eq=eq)


class ExternVar(DynVar):
    def __init__(self, name: str, init_eq: str, eq: str):
        DynVar.__init__(self, name=name,
                        init_eq=init_eq,
                        eq=eq)


class ExternState(ExternVar):
    def __init__(self, name: str, init_eq: str, eq: str):
        ExternVar.__init__(self, name=name,
                           init_eq=init_eq,
                           eq=eq)


class ExternAlgeb(ExternVar):
    def __init__(self, name: str, init_eq: str, eq: str):
        ExternVar.__init__(self, name=name,
                           init_eq=init_eq,
                           eq=eq)


class AliasState(ExternState):
    def __init__(self, name: str, init_eq: str, eq: str):
        ExternVar.__init__(self, name=name,
                           init_eq=init_eq,
                           eq=eq)
