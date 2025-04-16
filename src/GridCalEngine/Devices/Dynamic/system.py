# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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
        - Initialization of models and devices
        - Parsing of JSON configuration files
        - Create the DAE object for managing algebraic and differential equations
        - Setting up the system for simulation
        - Running time-domain simulations
    """

    def __init__(self):
        """
        Initializes the System instance.

        Attributes:
            models (dict):          A dictionary mapping model names to their respective instances.
            devices (dict):         A dictionary to store instantiated device objects.
            data (dict):            Parsed JSON data containing device configurations.
            models_list (list):     A dictionary of  abstract models to be processed.
            dae (DAE):              An instance of the DAE class for managing algebraic and differential equations.
            setup (SET):            An instance of the SET class for setting up the system.
            tds (TDS):              An instance of the TDS class for time-domain simulation.
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
        
        

