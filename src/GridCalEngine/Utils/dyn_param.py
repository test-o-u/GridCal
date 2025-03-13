# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Class to store params"""


class DynParam:
    def __init__(self, info: str, symbol: str):
        self.info = info
        self.symbol = symbol

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol


class NumDynParam(DynParam):
    def __init__(self, info: str, symbol: str, value: float):

        DynParam.__init__(self, symbol=symbol,
                          info=info)


class IdxDynParam(DynParam):
    def __init__(self, info: str, symbol: str):

        DynParam.__init__(self, symbol=symbol,
                          info=info)


class ExtParam(NumDynParam):
    def __init__(self, info: str, symbol: str):

        NumDynParam.__init__(self, symbol=symbol,
                             info=info)
