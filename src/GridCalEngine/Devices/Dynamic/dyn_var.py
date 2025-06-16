# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

'Class to store variables'

from typing import Dict, Any

from GridCalEngine.Devices.Dynamic.address import Address
from GridCalEngine.enumerations import DynamicVarType



class DynVar:
    def __init__(self, name: str, symbol: str, init_eq: str, eq: str, init_val: float = 0.0, src: str = "", indexer: str = ""):
        """

        :param name:
        :param symbol:
        :param init_eq
        :param eq:
        :param init_val:
        """
        self.name = name
        self.symbol = symbol
        self.src = src
        self.indexer = indexer
        self.init_eq = init_eq
        self.eq = eq
        self.init_val = init_val
        self.address: Address = Address()


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
            "src": self.src,
            "indexer": self.indexer,
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
        self.src = data["src"]
        self.indexer = data["indexer"]
        self.init_eq = data["init_eq"]
        self.eq = data["eq"]


class AlgebVar(DynVar):
    def __init__(self, name: str = "", symbol: str = "", src:str = "", indexer:str = "", init_eq: str = "", eq: str = "", init_val: float = 0.0):
        """

        :param name:
        :param symbol:
        :param init_eq:
        :param eq:
        :param init_val:
        """
        DynVar.__init__(self,
                        name=name,
                        symbol=symbol,
                        src=src,
                        indexer=indexer,
                        init_eq=init_eq,
                        eq=eq,
                        init_val=init_val)

        self.var_type: DynamicVarType = DynamicVarType.y

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        # d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        # self.var_type = data["var_type"]


class StatVar(DynVar):
    def __init__(self, name: str = "", symbol: str = "", src:str = "", indexer:str ="", init_eq: str = "", eq: str = "",
                 t_const: float = 1.0, init_val: float = 0.0):
        """

        :param name:
        :param symbol:
        :param init_eq:
        :param eq:
        :param t_const:
        :param init_val:
        """
        DynVar.__init__(self,
                        name=name,
                        symbol=symbol,
                        src=src,
                        indexer=indexer,
                        init_eq=init_eq,
                        eq=eq,
                        init_val=init_val)

        self.var_type: DynamicVarType = DynamicVarType.x
        self.t_const = t_const

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        # d["var_type"] = self.var_type
        # d["t_const"] = self.t_const
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        # self.var_type = data["var_type"]
        # self.t_const = data["t_const"]


# class InputVar(DynVar):
#     def __init__(self, name: str = "", symbol: str = "", src: str = "", indexer: str = "",
#                  init_eq: str = "", eq: str = "", init_val: float = 0.0):
#         """
#
#         :param name:
#         :param symbol:
#         :param src:
#         :param indexer:
#         :param init_eq:
#         :param eq:
#         :param init_val:
#         """
#         DynVar.__init__(self, name=name,
#                         symbol=symbol,
#                         init_eq=init_eq,
#                         eq=eq,
#                         init_val=init_val)
#         self.src = src
#         self.indexer = indexer
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Generates a json representation of this objects
#         :return: Dict[str, Any]
#         """
#         d = super().to_dict()
#         d["src"] = self.src
#         d["indexer"] = self.indexer
#         return d
#
#     def parse(self, data: Dict[str, Any]):
#         """
#         Parse jsn data generated with to_json
#         :param data: Dict[str, Any]
#         """
#         super().parse(data=data)
#         self.src = data["src"]
#         self.indexer = data["indexer"]
#

class InputState(StatVar):
    def __init__(self,
                 name: str = "",
                 symbol: str = "",
                 src: str = "",
                 indexer: str = "",
                 init_eq: str = "",
                 eq: str = "",
                 init_val: float = 0.0):
        """

        :param name:
        :param symbol:
        :param src:
        :param indexer:
        :param init_eq:
        :param eq:
        :param init_val:
        """
        StatVar.__init__(self, name=name,
                          symbol=symbol,
                          src=src,
                         indexer=indexer,
                          init_eq=init_eq,
                          eq=eq,
                          init_val=init_val)
        self.var_type: DynamicVarType = DynamicVarType.x

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        # d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        # self.var_type = data["var_type"]


class InputAlgeb(AlgebVar):
    def __init__(self, name: str = "", symbol: str = "", src: str = "",
                 indexer: str = "", init_eq: str = "", eq: str = "", init_val: float = 0.0):
        """

        :param name:
        :param symbol:
        :param src:
        :param indexer:
        :param init_eq:
        :param eq:
        :param init_val:
        """

        AlgebVar.__init__(self, name=name,
                          symbol=symbol,
                          src=src,
                          indexer=indexer,
                          init_eq=init_eq,
                          eq=eq,
                          init_val=init_val)
        self.var_type: DynamicVarType = DynamicVarType.y

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        # d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        # self.var_type = data["var_type"]




# class OutputVar(DynVar):
#     def __init__(self, name: str = "", symbol: str = "", src: str = "", indexer: str = "",
#                  init_eq: str = "", eq: str = "", init_val: float = 0.0):
#         """
#
#         :param name:
#         :param symbol:
#         :param src:
#         :param indexer:
#         :param init_eq:
#         :param eq:
#         :param init_val:
#         """
#         DynVar.__init__(self, name=name,
#                         symbol=symbol,
#                         init_eq=init_eq,
#                         eq=eq,
#                         init_val=init_val)
#         self.src = src
#         self.indexer = indexer
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Generates a json representation of this objects
#         :return: Dict[str, Any]
#         """
#         d = super().to_dict()
#         d["src"] = self.src
#         d["indexer"] = self.indexer
#         return d
#
#     def parse(self, data: Dict[str, Any]):
#         """
#         Parse jsn data generated with to_json
#         :param data: Dict[str, Any]
#         """
#         super().parse(data=data)
#         self.src = data["src"]
#         self.indexer = data["indexer"]


class OutputState(StatVar):
    def __init__(self, name: str = "", symbol: str = "", src: str = "", indexer: str = "", init_eq: str = "",
                 eq: str = "", init_val: float = 0.0):
        """

        :param name:
        :param symbol:
        :param src:
        :param indexer:
        :param init_eq:
        :param eq:
        :param init_val:
        """
        StatVar.__init__(self, name=name,
                          symbol=symbol,
                         src=src,
                         indexer=indexer,
                          init_eq=init_eq,
                          eq=eq,
                          init_val=init_val)
        self.var_type: DynamicVarType = DynamicVarType.x

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        # d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        # self.var_type = data["var_type"]


class OutputAlgeb(AlgebVar):
    def __init__(self, name: str = "", symbol: str = "", src: str = "",
                 indexer: str = "", init_eq: str = "", eq: str = "", init_val: float = 0.0):
        """

        :param name:
        :param symbol:
        :param src:
        :param indexer:
        :param init_eq:
        :param eq:
        :param init_val:
        """

        AlgebVar.__init__(self, name=name,
                          symbol=symbol,
                          src=src,
                          indexer=indexer,
                          init_eq=init_eq,
                          eq=eq,
                          init_val=init_val)
        self.var_type: DynamicVarType = DynamicVarType.y

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a json representation of this objects
        :return: Dict[str, Any]
        """
        d = super().to_dict()
        # d["var_type"] = self.var_type
        return d

    def parse(self, data: Dict[str, Any]):
        """
        Parse jsn data generated with to_json
        :param data: Dict[str, Any]
        """
        super().parse(data=data)
        # self.var_type = data["var_type"]
