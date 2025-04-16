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
    Represents a power system composed of various dynamic models and devices.

    Responsibilities include:
        - Loading configuration from JSON files
        - Initializing the DAE (Differential-Algebraic Equations) structure
        - Setting up devices and models
        - Running time-domain simulations
    """

    def __init__(self):
        """
        Initializes the System instance by loading configuration,
        creating required simulation components, and preparing for execution.

        Attributes:
            models (dict): Dictionary mapping model names to their instances.
            devices (dict): Dictionary of instantiated device objects.
            data (dict): Parsed JSON configuration data.
            models_list (list): List of abstract model types to be processed.
            dae (DAE): DAE system manager for equations.
            setup (SET): System setup handler.
            tds (TDS): Time-domain simulation engine.
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
        
        # Run time-domain simulation
        try:
            self.tds = TDS(self)
        except Exception as e:
            logging.info(f"An error occurred while simulating the TDS: {e}", exc_info=True)
        
        

