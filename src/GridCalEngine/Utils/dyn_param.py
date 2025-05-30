# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Class to store params"""


class DynParam:
    def __init__(self, info: str, symbol: str):
        self.info = info
        self.symbol = symbol

    #def __str__(self):
     #   return self.symbol

    #def __repr__(self):
     #   return self.symbol


class NumDynParam(DynParam):
    def __init__(self, info: str, name:str, symbol: str, value: float):

        DynParam.__init__(self, symbol=symbol,
                          info=info)
        self.name = name
        self.value=value


class IdxDynParam(DynParam):
    def __init__(self, info: str, name: str, symbol: str, id: list(), connection_point: object = str) -> None:

        DynParam.__init__(self, symbol=symbol,
                          info=info)
        self.name = name
        self.id=id
        self.connection_point = connection_point

class ExtDynParam(NumDynParam):
    def __init__(self, info: str, name: str, symbol: str, value: float):

        NumDynParam.__init__(self, info=info, name = name, symbol=symbol,
                              value = value)
        self.name = name
        self.value = value
