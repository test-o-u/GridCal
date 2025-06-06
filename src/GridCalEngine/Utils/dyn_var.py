# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

'Class to store variables'

from typing import Dict, Any


class DynVar:
    def __init__(self, name: str, symbol: str, init_eq: str, eq: str):
        """

        :param name:
        :param symbol:
        :param init_eq:
        :param eq:
        """
        self.name = name
        self.symbol = symbol
        self.init_eq = init_eq
        self.eq = eq

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        return {
            "name": self.name,
            "symbol": self.symbol,
            "init_eq": self.init_eq,
            "eq": self.eq
        }

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        self.name = data["name"]
        self.symbol = data["symbol"]
        self.init_eq = data["init_eq"]
        self.eq = data["eq"]


class AlgebVar(DynVar):
    def __init__(self, name: str = "", symbol: str = "", init_eq: str = "", eq: str = ""):
        """

        :param name:
        :param symbol:
        :param init_eq:
        :param eq:
        """
        DynVar.__init__(self,
                        name=name,
                        symbol=symbol,
                        init_eq=init_eq,
                        eq=eq)
        self.var_type = 'y'

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        #d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        #self.var_type = data["var_type"]


class StatVar(DynVar):
    def __init__(self, name: str = "", symbol: str = "", init_eq: str = "", eq: str = "", t_const=1.0):
        DynVar.__init__(self,
                        name=name,
                        symbol=symbol,
                        init_eq=init_eq,
                        eq=eq)
        self.var_type = 'x'
        self.t_const = t_const

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        #d["var_type"] = self.var_type
        #d["t_const"] = self.t_const
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        #self.var_type = data["var_type"]
        #self.t_const = data["t_const"]


class ExternVar(DynVar):
    def __init__(self, name: str = "", symbol: str = "", src: str = "", indexer: str = "",
                 init_eq: str = "", eq: str = ""):
        DynVar.__init__(self, name=name,
                        symbol=symbol,
                        init_eq=init_eq,
                        eq=eq)
        self.src = src
        self.indexer = indexer

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        d["src"] = self.src
        d["indexer"] = self.indexer
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        self.src = data["src"]
        self.indexer = data["indexer"]


class ExternState(ExternVar):
    def __init__(self,
                 name: str = "",
                 symbol: str = "",
                 src: str = "",
                 indexer: str = "",
                 init_eq: str = "",
                 eq: str = ""):
        ExternVar.__init__(self, name=name,
                           symbol=symbol,
                           src=src,
                           indexer=indexer,
                           init_eq=init_eq,
                           eq=eq)
        self.var_type = 'x'

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        #d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        #self.var_type = data["var_type"]


class ExternAlgeb(ExternVar):
    def __init__(self, name: str = "", symbol: str = "", src: str = "",
                 indexer: str = "", init_eq: str = "", eq: str = ""):
        """

        :param name:
        :param symbol:
        :param src:
        :param indexer:
        :param init_eq:
        :param eq:
        """

        ExternVar.__init__(self, name=name,
                           symbol=symbol,
                           src=src,
                           indexer=indexer,
                           init_eq=init_eq,
                           eq=eq)
        self.var_type = 'y'

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        #d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        #self.var_type = data["var_type"]
