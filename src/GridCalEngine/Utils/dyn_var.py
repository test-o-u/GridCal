# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

class DynVar:
    def __init__(self, name: str, eq: str):
        self.name = name
        self.eq = eq

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
