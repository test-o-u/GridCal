# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json

def readjson(input_file):
    """
    Parse the information in the json file.
    :param input_file: file with components data
    :return:
    """
    # Read json file and return a dict containing info for every component
    with open(input_file, "r", encoding="utf-8") as f:
        json_in = json.load(f)
        return json_in

