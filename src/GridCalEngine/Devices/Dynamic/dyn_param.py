# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Class to store params"""
from typing import List, Dict, Any


class DynParam:
    def __init__(self, info: str, symbol: str):
        self.info = info
        self.symbol = symbol

    # def __str__(self):
    #   return self.symbol

    # def __repr__(self):
    #   return self.symbol

    def to_dict(self) -> Dict[str, Any]:
        """

        :return:
        """
        return {
            "info": self.info,
            "symbol": self.symbol
        }

    def parse(self, data: Dict[str, Any]):
        """

        :param data:
        :return:
        """
        self.info = data["info"]
        self.symbol = data["symbol"]


class NumDynParam(DynParam):
    def __init__(self, info: str = "", name: str = "", symbol: str = "", value: float = 0):
        """

        :param info:
        :param name:
        :param symbol:
        :param value:
        """
        DynParam.__init__(self, symbol=symbol,
                          info=info)
        self.name = name
        self.value = value

    def to_dict(self):
        d = super().to_dict()
        d["name"] = self.name
        d["value"] = self.value
        return d

    def parse(self, data: Dict[str, Any]):
        """

        :param data:
        :return:
        """
        super().parse(data=data)
        self.name = data["name"]
        self.value = data["value"]


class IdxDynParam(DynParam):
    def __init__(self,
                 info: str = "",
                 name: str = "",
                 symbol: str = "",
                 ident: List[int] | None = None,
                 connection_point: str = "") -> None:
        """

        :param info:
        :param name:
        :param symbol:
        :param ident:
        :param connection_point:
        """
        DynParam.__init__(self, symbol=symbol,
                          info=info)
        self.name = name
        self.ident = ident
        self.connection_point = connection_point

    def to_dict(self):
        d = super().to_dict()
        d["name"] = self.name
        d["ident"] = self.ident
        d["connection_point"] = self.connection_point
        return d

    def parse(self, data: Dict[str, Any]):
        """

        :param data:
        :return:
        """
        super().parse(data=data)
        self.name = data["name"]
        self.ident = data["ident"]
        self.connection_point = data["connection_point"]


class ExtDynParam(NumDynParam):
    def __init__(self, info: str = "", name: str = "", symbol: str = "", value: float = 0.0):
        """

        :param info:
        :param name:
        :param symbol:
        :param value:
        """
        NumDynParam.__init__(self, info=info, name=name, symbol=symbol,
                             value=value)
        self.name = name
        self.value = value

    def to_dict(self):
        d = super().to_dict()
        d["name"] = self.name
        d["value"] = self.value
        return d

    def parse(self, data: Dict[str, Any]):
        """

        :param data:
        :return:
        """
        super().parse(data=data)
        self.name = data["name"]
        self.value = data["value"]
