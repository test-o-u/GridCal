# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import List, Dict, Any
from GridCalEngine.Devices.Parents.editable_device import EditableDevice, DeviceType
from GridCalEngine.Devices.Dynamic.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb
from GridCalEngine.Devices.Dynamic.dyn_param import NumDynParam, IdxDynParam, ExtDynParam


class DynamicModel(EditableDevice):
    "This class contains the parameters and variables needed to construct a dynamic model"

    def __init__(self, name: str = "", idtag: str | None = None):
        """
        DynamicModel class constructor
        :param name: Name of the Model
        """
        super().__init__(name=name,
                         idtag=idtag,
                         code="",
                         device_type=DeviceType.DynModel, )

        # connexion status
        self.u: List[int] = list()

        self.comp_name: List[str] = list()
        self.comp_code: List[int] = list()

        self.idx_dyn_param: Dict[str, IdxDynParam] = dict()
        self.num_dyn_param: Dict[str, NumDynParam] = dict()
        self.ext_dyn_param: Dict[str, ExtDynParam] = dict()

        self.stat_var: Dict[str, StatVar] = dict()
        self.algeb_var: Dict[str, AlgebVar] = dict()
        self.ext_state_var: Dict[str, ExternState] = dict()
        self.ext_algeb_var: Dict[str, ExternAlgeb] = dict()

    @property
    def idtag(self):
        return self._idtag

    @idtag.setter
    def idtag(self, val: str):
        self._idtag = val

    def get_var_num(self) -> int:
        """
        Get the number of variables
        :return:
        """
        return len(self.stat_var) + len(self.algeb_var) + len(self.ext_state_var) + len(self.ext_algeb_var)

    @property
    def is_empty(self) -> bool:
        return self.get_var_num() == 0

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a dictionary representation of this object
        :return: Dictionary
        """
        return {
            "idtag": self.idtag,
            "idx_dyn_param": [e.to_dict() for _, e in self.idx_dyn_param.items()],
            "num_dyn_param": [e.to_dict() for _, e in self.num_dyn_param.items()],
            "ext_dyn_param": [e.to_dict() for _, e in self.ext_dyn_param.items()],
            "stat_var": [e.to_dict() for _, e in self.stat_var.items()],
            "algeb_var": [e.to_dict() for _, e in self.algeb_var.items()],
            "ext_state_var": [e.to_dict() for _, e in self.ext_state_var.items()],
            "ext_algeb_var": [e.to_dict() for _, e in self.ext_algeb_var.items()],

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
        self.name: str = data["name"]
        self.idtag: str = data["idtag"]

        # connexion status
        self.u: List[int] = data["u"]
        self.comp_name: List[str] = data["comp_name"]
        self.comp_code: List[int] = data["comp_code"]

        self.idx_dyn_param.clear()
        for elm in data["idx_dyn_param"]:
            obj = IdxDynParam()
            obj.parse(data=elm)
            self.add_idx_dyn_param(obj)

        self.num_dyn_param.clear()
        for elm in data["num_dyn_param"]:
            obj = NumDynParam()
            obj.parse(data=elm)
            self.add_num_dyn_param(obj)

        self.ext_dyn_param.clear()
        for elm in data["ext_dyn_param"]:
            obj = ExtDynParam()
            obj.parse(data=elm)
            self.add_ext_dyn_param(obj)

        self.stat_var.clear()
        for elm in data["stat_var"]:
            obj = StatVar()
            obj.parse(data=elm)
            self.add_stat_var(obj)

        self.algeb_var.clear()
        for elm in data["algeb_var"]:
            obj = AlgebVar()
            obj.parse(data=elm)
            self.add_algeb_var(obj)

        self.ext_state_var.clear()
        for elm in data["ext_state_var"]:
            obj = ExternState()
            obj.parse(data=elm)
            self.add_ext_state_var(obj)

        self.ext_algeb_var.clear()
        for elm in data["ext_algeb_var"]:
            obj = ExternAlgeb()
            obj.parse(data=elm)
            self.add_ext_algeb_var(obj)

    def add_idx_dyn_param(self, val: IdxDynParam):
        self.idx_dyn_param[val.name] = val

    def add_num_dyn_param(self, val: NumDynParam):
        self.num_dyn_param[val.name] = val

    def add_ext_dyn_param(self, val: ExtDynParam):
        self.ext_dyn_param[val.name] = val

    def add_stat_var(self, val: StatVar):
        self.stat_var[val.name] = val

    def add_algeb_var(self, val: AlgebVar):
        self.algeb_var[val.name] = val

    def add_ext_state_var(self, val: ExternState):
        self.ext_state_var[val.name] = val

    def add_ext_algeb_var(self, val: ExternAlgeb):
        self.ext_algeb_var[val.name] = val

    def get_idx_dyn_param(self, val: str):
        return self.idx_dyn_param[val]

    def get_num_dyn_param(self, val: str):
        return self.num_dyn_param[val]

    def get_ext_dyn_param(self, val: str):
        return self.ext_dyn_param[val]

    def get_stat_var(self, val: str):
        return self.stat_var[val]

    def get_algeb_var(self, val: str):
        return self.algeb_var[val]

    def get_ext_state_var(self, val: str):
        return self.ext_state_var[val]

    def get_ext_algeb_var(self, val: str):
        return self.ext_algeb_var[val]
