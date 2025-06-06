# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

'Class to store services'


class DynService:
    def __init__(self, name: str, symbol: str, init_eq: str, eq: str):
        self.name = name
        self.symbol = symbol
        self.init_eq = init_eq
        self.eq = eq

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

