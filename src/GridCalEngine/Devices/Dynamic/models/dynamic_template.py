# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from GridCalEngine.Devices.Parents.editable_device import EditableDevice, DeviceType
from typing import Union


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
