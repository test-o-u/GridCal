# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Dict
from uuid import uuid4
import sympy as smb
from GridCalEngine.Devices.Parents.editable_device import EditableDevice


class Var:

    def __init__(self, name: str="", parent: EditableDevice | None = None):
        """

        :param name:
        :param parent:
        """
        self.idtag: int = uuid4().int

        self.name = name

        self.symbol = smb.Symbol(name)

        self.parent: EditableDevice | None = parent

    def __str__(self):
        return self.symbol.name

    def to_dict(self) -> Dict[str, str | int]:
        """
        Generate dictionary with the variable info
        :return: dictionary[name, value]
        """
        return {
            "idtag": self.idtag,
            "name": self.name,
            "parent": self.parent
        }

    def parse(self, data: Dict[str, str | int]):
        """
        Parse information saved with self.to_dict()
        :param data:
        :return:
        """
        self.idtag = data["idtag"]
        self.name = data["name"]
        self.parent = data.get("parent", None)
        self.symbol = smb.Symbol(self.name)