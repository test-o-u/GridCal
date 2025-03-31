# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import time
from GridCalEngine.Devices.Dynamic.system import System
from GridCalEngine.Devices.Dynamic.model_list import MODELS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# NOTE: Other tests
# 'GridCalEngine/Devices/Dynamic/test.json'
# 'GridCalEngine/Devices/Dynamic/test_2buses1line.json'
# 'GridCalEngine/Devices/Dynamic/test_3buses3lines.json'
datafile = 'GridCalEngine/Devices/Dynamic/test.json'

def main():
    """
    Main function to initialize and run the system simulation.
    """
    try:
        start_time = time.perf_counter()

        # Initialize the system with given models and datafile
        system = System(MODELS, datafile)

        # Performance timing logs
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
        logging.info(f"GENCLS a = {system.models['GENCLS'].algeb_idx}")
        logging.info(f"GENCLS a = {system.models['GENCLS'].extalgeb_idx}")
        logging.info("===========================================")

        logging.info("=============== JACOBIANS ================")
        logging.info(f"dgy = {system.dae.dgy}")
        logging.info("===========================================")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
