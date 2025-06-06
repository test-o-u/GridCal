# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
from GridCalEngine.Devices.Parents.editable_device import EditableDevice
from GridCalEngine.enumerations import DeviceType
from GridCalEngine.Devices.Dynamic.dyn_var import AliasState


class Unlock(EditableDevice):
    "This class contains the information needed to check if a device connected to a Bus is trustable"


    def __init__(self,
                 name: str,
                 code: str,
                 idtag: Union[str, None]):
        """
        Unlock class constructor
        :param name: Name
        :param code: secondary code
        :param idtag: UUID code
        """
        
        EditableDevice.__init__(self, name, code, idtag, device_type=DeviceType.DynSynchronousModel)

        self.Outputs_dict = {'GENCLS': list(),
                             'ACLine': list(),
                             'ExpLoad': list()}
        self.Inputs_dict = {'GENCLS': list(),
                             'ACLine': list(),
                             'ExpLoad': list()}