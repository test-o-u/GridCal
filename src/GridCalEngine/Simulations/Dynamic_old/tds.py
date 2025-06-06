# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Devices.Dynamic.integration import method_map
from GridCalEngine.Devices.Dynamic.utils.data_processing import DataProcessor


class TDS():
    """
    Time domain simulation class.
    """

    def __init__(self, system):
        """
        TDS constructor
        Initializes and executes time domain simulation.
        :param system: The system instance containing models and devices
        """
        # Pass the system object
        self.system = system

        # Set simulation parameters
        self.dt = config.TIME_STEP  # Time step for integration
        self.t_final = config.SIMULATION_TIME  # Final simulation time
        self.method_tds = config.INTEGRATION_METHOD  # Integration method for time domain simulation
        self.method_ss = config.STEADYSTATE_METHOD  # Steady-state method for simulation

        # Initialize results list
        self.results = []  # List to store simulation results

        # Initialize data processor
        self.dataprocessor = DataProcessor(self.system)  # Data processor to evaluate simulation data

        # Get integration method
        if self.method_tds not in method_map or self.method_ss not in method_map:
            raise ValueError(f"Unknown integration method: {self.method_tds}")
        self.integrator = method_map[self.method_tds]  # The integration method object
        self.steadystate = method_map[self.method_ss]  # The steady-state method object

        # Time domain simulation
        # Initialize simulatoin
        self.system.dae.initilize_fg()
        # Run simulation
        #self.run_steadystate()
        self.run_tds()
        self.save_simulation_data()

    def run_tds(self):
        """
        Performs the numerical integration using the chosen method.
        :return:
        """
        t = 0
        while t < self.t_final:
            # Solve DAE step
            converged = self.integrator.step(dae=self.system.dae, dt=self.dt, method=self.integrator, tol=config.TOL,
                                             max_iter=config.MAX_ITER)

            if not converged:
                raise RuntimeError("Integration step did not converge.")

            t += self.dt
            self.results.append((t, self.system.dae.x.copy(), self.system.dae.y.copy()))

    def run_steadystate(self):
        """
        Performs steady-state computation.
        :return:
        """
        converged = self.steadystate.steadystate(dae=self.system.dae, method=self.steadystate, tol=config.TOL,
                                                 max_iter=config.MAX_ITER)

        if converged:
            print(f"Steady-state found.")
        else:
            raise RuntimeError("Steady-state not found.")

    def save_simulation_data(self):
        """
        Processes simulation data
        :return:
        """
        self.dataprocessor.save_data(self.results)
        self.dataprocessor.export_csv()
        self.dataprocessor.plot_results()
