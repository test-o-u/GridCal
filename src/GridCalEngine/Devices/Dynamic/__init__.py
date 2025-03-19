# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import importlib
import io
import logging
import os
import chardet
from GridCalEngine.Devices.Dynamic.io.json import readjson
from GridCalEngine.Devices.Dynamic.system import System
from GridCalEngine.Devices.Dynamic.model_list import model_list
from GridCalEngine.Devices.Dynamic.models.dynamic_model_template import DynamicModelTemplate
inputfile_path = 'GridCalEngine/Devices/Dynamic/tryout.json'

def prepare(inputfile_path):
    my_system = System(model_list)
    my_system.import_models()
    components_info = readjson(inputfile_path)
    my_system.prepare()



prepare(inputfile_path)
