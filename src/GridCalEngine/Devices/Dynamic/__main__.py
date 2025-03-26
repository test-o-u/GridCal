# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Dynamic simulator entry point. To run the _main_ module of the Dynamic package:
    cd src
    python -m GridCalEngine.Devices.Dynamic
"""

from GridCalEngine.Devices.Dynamic import main


# we need to create a cli and define a main() function that calls the functions defined in main.py
#if __name__ == '__main__':
    #main()
