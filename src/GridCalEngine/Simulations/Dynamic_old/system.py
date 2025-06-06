# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from collections import OrderedDict
from GridCalEngine.Simulations.Dynamic.problems.rms_problem import DAE
from GridCalEngine.Simulations.Dynamic.set import SET
from GridCalEngine.Simulations.Dynamic.tds import TDS



class System:
    """
    Represents a power system composed of various dynamic models and devices.

    Responsibilities include:
        - Loading configuration from JSON files
        - Initializing the DAE (Differential-Algebraic Equations) structure
        - Setting up devices and models
        - Running time-domain simulations
    """

    def __init__(self, dynamic_grid):
        """
        System class constructor.
        Initializes the System instance by loading configuration,
        creating required simulation components, and preparing for execution.
        """

        self.models = {}  # Dictionary mapping model names to their instances.
        self.devices = OrderedDict()  # Dictionary of instantiated device objects.
        self.connections = list() # List of well-connected elements.
        self.devices_list = dynamic_grid

        # Instanciate DAE object
        try:
            self.dae = DAE(self)  # DAE system manager for equations.
        except Exception as e:
            logging.info(f"An error occurred while initializing the DAE: {e}", exc_info=True)

        # Setup the system
        try:
            self.setup = SET(self, self.devices_list)  # System setup handler.
        except Exception as e:
            logging.info(f"An error occurred while setting-up the SET: {e}", exc_info=True)

        # Run time-domain simulation
        try:
            self.tds = TDS(self)  # Time-domain simulation engine.
        except Exception as e:
            logging.info(f"An error occurred while simulating the TDS: {e}", exc_info=True)
