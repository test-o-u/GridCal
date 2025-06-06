# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Simulations.Dynamic.integration import method_map
from GridCalEngine.Simulations.Dynamic.utils.data_processing import DataProcessor
from GridCalEngine.Simulations.Dynamic.dynamic_system_store import DynamicSystemStore
from GridCalEngine.Simulations.Dynamic.integration import BackEuler, Trapezoid, SteadyState
from GridCalEngine.enumerations import DynamicIntegrationMethod


class TimeDomainSimulation:
    """
    Time domain simulation class.
    """

    def __init__(self, dynamic_system_store: DynamicSystemStore,
                 time_step: float,
                 simulation_time: float,
                 max_integrator_iterations: int,
                 tolerance: float,
                 integration_method: DynamicIntegrationMethod = DynamicIntegrationMethod.Trapezoid):
        """
        TDS constructor
        Initializes and executes time domain simulation.
        :param dynamic_system_store: The system instance containing models and devices
        """
        # Pass the system object
        self.dynamic_system_store = dynamic_system_store

        # Set simulation parameters
        self.dt = time_step  # Time step for integration
        self.t_final = simulation_time  # Final simulation time
        self.integration_method = integration_method  # Integration method for time domain simulation
        self.max_integrator_iterations = max_integrator_iterations
        self.tolerance = tolerance

        # Initialize results list
        self.results = []  # List to store simulation results

        # Initialize data processor
        self.dataprocessor = DataProcessor(self.dynamic_system_store)  # Data processor to evaluate simulation data

    def run(self):
        """
        Performs the numerical integration using the chosen method.
        :return:
        """
        # Time domain simulation
        self.dynamic_system_store.dae.initilize_fg()

        # Get integration method
        if self.integration_method == DynamicIntegrationMethod.Trapezoid:
            integrator = Trapezoid()
        elif self.integration_method == DynamicIntegrationMethod.BackEuler:
            integrator = BackEuler()
        else:
            raise ValueError(f"integrator not implemented :( {self.integration_method}")

        t = 0
        while t < self.t_final:
            # Solve DAE step
            converged = integrator.step(dae=self.dynamic_system_store.dae,
                                        dt=self.dt,
                                        method=self.integrator,
                                        tol=self.tolerance,
                                        max_iter=self.max_integrator_iterations)

            if not converged:
                raise RuntimeError("Integration step did not converge.")

            t += self.dt
            self.results.append(
                (t,
                 self.dynamic_system_store.dae.x.copy(),
                 self.dynamic_system_store.dae.y.copy())
            )

    def save_simulation_data(self):
        """
        Processes simulation data
        :return:
        """
        self.dataprocessor.save_data(self.results)
        self.dataprocessor.export_csv()
        self.dataprocessor.plot_results()
