# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Utils.dyn_var import StatVar, AlgebVar, ExternState, ExternAlgeb, DynVar
from GridCalEngine.Utils.dyn_param import NumDynParam, IdxDynParam, ExtDynParam


class DynamicModel(DynamicModelTemplate):
    "This class contains the parameters and variables needed to construct a dynamic model"

    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
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
        self.u = list()

        self.comp_name = list()
        self.comp_code = list()
