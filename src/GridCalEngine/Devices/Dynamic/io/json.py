# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json
from GridCalEngine.Devices.Dynamic.system import System


def read(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        json_in = json.load(f)
    for model_name, dct in json_in.items():
        for component_info in dct:
            System.add_components(model_name, component_info)
