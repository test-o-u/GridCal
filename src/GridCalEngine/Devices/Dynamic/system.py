# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import time
import logging
import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Devices.Dynamic.dae import DAE
from GridCalEngine.Devices.Dynamic.set import SET
from GridCalEngine.Devices.Dynamic.tds import TDS
from GridCalEngine.Devices.Dynamic.utils.json import readjson

class System:
    """
    This class represents a power system containing various models and devices.

    It handles:
    - Importing and managing abstract dynamic models.
    - Processing symbolic and numerical computations.
    - Creating and managing device instances in one single object in a vecotrized form.
    - Assigning global indices to system variables and copies of these to external system variables.
    """

    def __init__(self):
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
        # Initialize empty attributes
        self.models = {}
        self.devices = {}

        # Parse the JSON data file
        self.data = readjson(config.SYSTEM_JSON_PATH)
        # Get the list of models from the config file
        self.models_list = config.MODELS

        # Instanciate DAE object
        try:
            self.dae = DAE(self)
        except Exception as e:
            logging.info(f"An error occurred while initializing the DAE: {e}", exc_info=True)
        
        # Setup the system
        try:
            self.setup = SET(self, self.models_list, self.data)
        except Exception as e:
            logging.info(f"An error occurred while setting-up the SET: {e}", exc_info=True)
        
        # Simulate the system
        try:
            self.tds = TDS(self)
        except Exception as e:
            logging.info(f"An error occurred while simulating the TDS: {e}", exc_info=True)
        
        

