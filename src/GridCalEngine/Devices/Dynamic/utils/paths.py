# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os

def get_generated_module_path(mkdir=True):
    """
    Get the path to the ``generated_module`` folder.
    """


    generated_module_path = 'GridCalEngine/Devices/Dynamic/generated_module'



    if mkdir is True:
        os.makedirs(generated_module_path, exist_ok=True)

    return generated_module_path
