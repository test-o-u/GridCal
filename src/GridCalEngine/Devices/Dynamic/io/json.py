# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json
def read(input_file):
    json_in = json.load(input_file)
    for name, dct in json_in.items():
        for row in dct:
            system.add(name, row)