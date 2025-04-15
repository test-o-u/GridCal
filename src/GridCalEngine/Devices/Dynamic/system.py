# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from GridCalEngine.Devices.Dynamic.dae import DAE
from GridCalEngine.Devices.Dynamic.setup import Setup
from GridCalEngine.Devices.Dynamic.utils.paths import get_generated_module_path
from GridCalEngine.Devices.Dynamic.io.json import readjson


class System:
    """
    This class represents a power system containing various models and devices.

    It handles:
    - Importing and managing abstract dynamic models.
    - Processing symbolic and numerical computations.
    - Creating and managing device instances in one single object in a vecotrized form.
    - Assigning global indices to system variables and copies of these to external system variables.
    """

    def __init__(self, models_list, datafile):
        """
        Initializes the System instance.

        Args:
            models_list (list): A list of model categories and their associated models.
            datafile (str): Path to the JSON file containing device data and system configuration.

        Attributes:
            models_list (list): Stores the provided list of model categories and models.
            models (dict): A dictionary mapping model names to their respective instances.
            devices (dict): A dictionary to store instantiated device objects.
            dae (DAE): An instance of the DAE class for managing algebraic and differential equations.
            data (dict): Parsed JSON data containing device configurations.
        """

        self.models_list = models_list
        self.values_array = None
        self.data = readjson(datafile)
        self.models = {}
        self.devices = {}
        self.dae = DAE(self)
        self.setup = Setup(self, models_list, datafile)
