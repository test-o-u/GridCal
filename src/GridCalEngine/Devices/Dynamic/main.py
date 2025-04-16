# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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
import time
import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Devices.Dynamic.system import System

### Configure logging ###
logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    """
    Main function to initialize and run the system simulation.
    """
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
    start_time = time.perf_counter()
    try:
        # Instanciate the system
        system = System()

    except Exception as e:
        logging.error(f"An error occurred while initializing the system: {e}", exc_info=True)

    # Performance timing logs
    if config.PERFORMANCE:
        logging.info("=============== TIME CHECK ================")
        logging.info(f"Process symbolic time = {system.symb_time:.6f} [s]")
        logging.info(f"Create device time = {system.dev_time:.6f} [s]")
        logging.info(f"Set address time = {system.add_time:.6f} [s]")
        total_time = time.perf_counter() - start_time
        logging.info(f"Total execution time: {total_time:.6f} [s]")
        logging.info("===========================================")
        logging.info("=============== ADDRESS CHECK =============")
        logging.info(f"Bus a = {system.models['Bus'].algeb_idx}")
        logging.info(f"ACLine a = {system.models['ACLine'].extalgeb_idx}")
        logging.info(f"ExpLoad a = {system.models['ExpLoad'].extalgeb_idx}")
        logging.info(f"GENCLS a = {system.models['GENCLS'].states_idx}")
        logging.info(f"GENCLS a = {system.models['GENCLS'].algeb_idx}")
        logging.info(f"GENCLS a = {system.models['GENCLS'].extalgeb_idx}")
        logging.info("===========================================") 

if __name__ == "__main__":
    main()