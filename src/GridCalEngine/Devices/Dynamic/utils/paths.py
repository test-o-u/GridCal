# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os

def get_pycode_path(pycode_path=None, mkdir=True):
    """
    Get the path to the ``pycode`` folder.
    """

    if pycode_path is None:
        pycode_path = 'GridCalEngine/Devices/Dynamic/pycode'


    if mkdir is True:
        os.makedirs(pycode_path, exist_ok=True)

    return pycode_path
