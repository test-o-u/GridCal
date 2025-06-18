# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from uuid import uuid4
import sympy as smb
from GridCalEngine.Devices.Parents.editable_device import EditableDevice


class Var:

    def __init__(self, name: str, parent: EditableDevice | None = None):
        """

        :param name:
        :param parent:
        """
        self.idtag: int = uuid4().int

        self.symbol = smb.Symbol(name)

        self.parent: EditableDevice | None = parent

    def __str__(self):
        return self.symbol.name
