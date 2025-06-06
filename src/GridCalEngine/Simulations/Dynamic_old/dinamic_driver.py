# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import pdb

from GridCalEngine.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Simulations.Dynamic.system import System
from GridCalEngine.Simulations.Dynamic.utils.logging_config import setup_logging
from GridCalEngine.Simulations.driver_template import DriverTemplate
from GridCalEngine.enumerations import EngineType, SimulationTypes

class DynamicDriver(DriverTemplate):
    name = 'Power Flow'
    tpe = SimulationTypes.RmsDynamic_run

    """
    Dynamic wrapper to use with Qt
    """

    def __init__(self, grid: MultiCircuit, engine: EngineType = EngineType.GridCal):

    # to use when options and results are defined
    # def __init__(self, grid: MultiCircuit,
    #              options: Union[Dynamic_options, None] = None,
    #              opf_results: Union[OptimalPowerFlowResults, None] = None,
    #              engine: EngineType = EngineType.GridCal):



        """
        DynamicDriver class constructor
        :param grid: MultiCircuit instance
        :param options: PowerFlowOptions instance (optional)
        :param opf_results: OptimalPowerFlowResults instance (optional)
        :param engine: EngineType (i.e., EngineType.GridCal) (optional)
        """

        DriverTemplate.__init__(self, grid=grid, engine=engine)


    def run(self):
        """
        Main function to initialize and run the system simulation.

        This function sets up logging, starts the dynamic simulation, and
        logs the outcome. It handles and logs any exceptions raised during execution.
        :return:
        """
        # Set up logging
        setup_logging()

        try:
            # Run the dynamic simulation
            self.start_dynamic()
            logging.info("Simulation completed successfully.")

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

    def start_dynamic(self):
        """
        Initializes the dynamic system.

        This function instantiates the System object required to start the
        dynamic simulation. Logs any exceptions raised during initialization.
        :return:
        """
        try:
            #   Extract dynamic devices from grid
            dynamic_devices = [device._dynamic_model for device in self.grid.items()]
            #  Instantiate the system
            System(dynamic_devices)

        except Exception as e:
            logging.error(
                f"An error occurred while initializing the system: {e}",
                exc_info=True
            )



