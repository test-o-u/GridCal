# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from GridCalEngine.Devices.Parents.editable_device import EditableDevice, DeviceType
from typing import Union
from GridCalEngine.Devices.Dynamic.spoint import Spoint
from GridCalEngine.Devices.Dynamic.symprocess import Symprocess
from GridCalEngine.Utils.dyn_param import NumDynParam
from GridCalEngine.Utils.dyn_var import *
from GridCalEngine.Utils.dyn_param import *



class DynamicModelTemplate(EditableDevice):

    def __init__(self, name: str, code: str, idtag: Union[str, None],
                 device_type: DeviceType):
        """

        :param name:
        :param code:
        :param idtag:
        :param device_type:
        """
        EditableDevice.__init__(self,
                                name=name,
                                code=code,
                                idtag=idtag,
                                device_type=device_type)

        self.spoint = Spoint(self.name)
        self.dict = self.__dict__
        self.symp = Symprocess(self)

        # Set address function
        self.n = 0
        self.algeb_idx = {}
        self.extalgeb_idx = {}

    def store_data(self):

        for key, elem in self.dict.items():

            if isinstance(elem, AlgebVar):
                self.spoint.add_algebvars(elem)
            if isinstance(elem, StatVar):
                self.spoint.add_statvars(elem)
            if isinstance(elem, ExternVar):
                self.spoint.add_externvars(elem)
            if isinstance(elem, ExternState):
                self.spoint.add_externstates(elem)
            if isinstance(elem, ExternAlgeb):
                self.spoint.add_externalgebs(elem)
            if isinstance(elem, AliasAlgeb):
                self.spoint.add_aliasalgebs(elem)
            if isinstance(elem, AliasState):
                self.spoint.add_aliastats(elem)

            if isinstance(elem, NumDynParam):
                self.spoint.add_numdynparam(elem)
            if isinstance(elem, IdxDynParam):
                self.spoint.add_idxdynparam(elem)
            if isinstance(elem, ExtParam):
                self.spoint.add_extparam(elem)

    def process_data(self):
        self.symp.generate()
         
