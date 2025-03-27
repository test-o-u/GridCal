# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from GridCalEngine.Devices.Dynamic.system import System
from GridCalEngine.Devices.Dynamic.model_list import MODELS
from GridCalEngine.Devices.Dynamic.io.json import readjson

# NOTE: Other tests
# 'GridCalEngine/Devices/Dynamic/test.json'
# 'GridCalEngine/Devices/Dynamic/test_2buses1line.json'
# 'GridCalEngine/Devices/Dynamic/test_3buses3lines.json'
datafile = 'GridCalEngine/Devices/Dynamic/test_3buses3lines.json'

def main():
    # Initialize the abstract system components
    system = System(MODELS, datafile)

    print("=============== TIME CHECK ================")
    print(f"Process symbolic time = {system.symb_time} [s]")
    print(f"Process create device time = {system.dev_time} [s]")
    print(f"Process set address time = {system.add_time} [s]")
    print("===========================================")

    print("=============== ADDRESS CHECK ================")
    print(f"Bus a = {system.models['Bus'].algeb_idx['a']}")
    print(f"ACLine a = {system.models['ACLine'].extalgeb_idx['a']}")
    print("==============================================")