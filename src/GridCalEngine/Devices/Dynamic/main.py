# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# TODO: ask Santiago what's the best practice 
import os
import sys

# Dynamically add the project src/ folder to sys.path
CURRENT_FILE = os.path.abspath(__file__)
# Change module import path to the src/ folder
SRC_PATH = os.path.abspath(os.path.join(CURRENT_FILE, "../../../../"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# Change the current working directory to the src/ folder
os.chdir(SRC_PATH)

import logging
from GridCalEngine.Devices.Dynamic.system import System
from GridCalEngine.Devices.Dynamic.utils.logging_config import setup_logging

def main():
    """
    Main function to initialize and run the system simulation.
    """

    # Set up logging
    setup_logging()

    try:
        # Run the dynamic simulation
        start_dynamic()
        logging.info("Simulation completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
 
def start_dynamic():
    """
    System initialization function.
    """
    
    try:
        # Instanciate the system
        system = System()

    except Exception as e:
        logging.error(f"An error occurred while initializing the system: {e}", exc_info=True)

if __name__ == "__main__":
    main()