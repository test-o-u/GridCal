# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List, Dict, Any
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam, ExtDynParam


class DynamicModel(DynamicModelTemplate):
    "This class contains the parameters and variables needed to construct a dynamic model"

    def __init__(self,
                 name: str = "",
                 code: str = "",
                 idtag: Union[str, None] = ""):
        """
        DynamicModel class constructor
        :param name: Name of the GENCLS
        :param code: secondary code
        :param idtag: UUID code
        """

        DynamicModelTemplate.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

        # to fill all the params and variables above from json file
        self.idx_dyn_param: List[IdxDynParam] = list()
        self.num_dyn_param: List[NumDynParam] = list()
        self.ext_dyn_param: List[ExtDynParam] = list()
        self.stat_var: List[StatVar] = list()
        self.algeb_var: List[AlgebVar] = list()
        self.ext_state_var: List[ExternState] = list()
        self.ext_algeb_var: List[ExternAlgeb] = list()

        # connexion status
        self.u: List[int] = list()

        self.comp_name: List[str] = list()
        self.comp_code: List[int] = list()

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a dictionary representation of this object
        :return: Dictionary
        """
        return {
            "idx_dyn_param": [e.to_dict() for e in self.idx_dyn_param],
            "num_dyn_param": [e.to_dict() for e in self.num_dyn_param],
            "ext_dyn_param": [e.to_dict() for e in self.ext_dyn_param],
            "stat_var": [e.to_dict() for e in self.stat_var],
            "algeb_var": [e.to_dict() for e in self.algeb_var],
            "ext_state_var": [e.to_dict() for e in self.ext_state_var],
            "ext_algeb_var": [e.to_dict() for e in self.ext_algeb_var],

            "u": self.u,

            "comp_name": self.comp_name,
            "comp_code": self.comp_code
        }

    def parse(self, data: Dict[str, List[Dict[str, Any]]] | List[int] | List[str]):
        """
        Parse the dictionary representation of this object
        :param data:
        :return:
        """
        self.idx_dyn_param: List[IdxDynParam] = list()
        for elm in data["idx_dyn_param"]:
            obj = IdxDynParam()
            obj.parse(data=elm)
            self.idx_dyn_param.append(obj)


        self.num_dyn_param: List[NumDynParam] = list()
        for elm in data["num_dyn_param"]:
            obj = NumDynParam()
            obj.parse(data=elm)
            self.num_dyn_param.append(obj)


        self.ext_dyn_param: List[ExtDynParam] = list()
        for elm in data["ext_dyn_param"]:
            obj = ExtDynParam()
            obj.parse(data=elm)
            self.ext_dyn_param.append(obj)

        self.stat_var: List[StatVar] = list()
        for elm in data["stat_var"]:
            obj = StatVar()
            obj.parse(data=elm)
            self.stat_var.append(obj)

        self.algeb_var: List[AlgebVar] = list()
        for elm in data["algeb_var"]:
            obj = AlgebVar()
            obj.parse(data=elm)
            self.algeb_var.append(obj)

        self.ext_state_var: List[ExternState] = list()
        for elm in data["ext_state_var"]:
            obj = ExternState()
            obj.parse(data=elm)
            self.ext_state_var.append(obj)

        self.ext_algeb_var: List[ExternAlgeb] = list()
        for elm in data["ext_algeb_var"]:
            obj = ExternAlgeb()
            obj.parse(data=elm)
            self.ext_algeb_var.append(obj)

        # connexion status
        self.u: List[int] = data["u"]

        self.comp_name: List[str] = data["comp_name"]
        self.comp_code: List[int] = data["comp_code"]