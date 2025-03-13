# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import importlib
import io
import logging
import os
import chardet
from system import import_models
from io.json import read

input_formats = {
    'xlsx': ('xlsx',),
    'json': ('json',),
    'matpower': ('m', ),
    'psse': ('raw', 'dyr'),
}
def parse(inputfile_path):
    read(inputfile_path)


