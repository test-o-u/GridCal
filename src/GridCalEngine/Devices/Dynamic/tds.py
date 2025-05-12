# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Utils.dyn_param import NumDynParam
from GridCalEngine.Devices.Dynamic.integration import method_map
from GridCalEngine.Devices.Dynamic.utils.data_processing import Data_processor
from GridCalEngine.Devices.Dynamic.utils.json import readjson


class TDS():
    """
    Time domain simulation class.
    """
    
    def __init__(self, system):
        """
        Initializes and executes time domain simulation.

        Attributes:
            system (System):            The system instance containing models and devices.
            dt (float):                 Time step for integration.
            t_final (float):            Final simulation time.
            method_tds (str):           Integration method for time domain simulation.
            method_ss (str):            Steady-state method for simulation.
            results (list):             List to store simulation results.
            integrator (Integrator):    The integration method object.
            steadystate (SteadyState):  The steady-state method object.
        """
        # Pass the system object
        self.system = system

        # Load events from JSON file
        self.events = readjson(config.EVENTS_JSON_PATH)
        self.applied_events = [] 

        # Set simulation parameters
        self.t = 0.0
        self.dt = config.TIME_STEP
        self.t_final = config.SIMULATION_TIME
        self.method_tds = config.INTEGRATION_METHOD
        self.method_ss = config.STEADYSTATE_METHOD

        # Initialize results list
        self.results = []

        # Save simulation data
        self.data_processor = Data_processor(self.system)

        # Get integration method
        if self.method_tds not in method_map or self.method_ss not in method_map:
            raise ValueError(f"Unknown integration method: {self.method_tds}")
        self.integrator = method_map[self.method_tds]
        self.steadystate = method_map[self.method_ss]

        # Time domain simulation
        # Initialize simulatoin
        self.system.dae.initilize_fg()
        # Run simulation
        #self.run_steadystate()
        self.run_tds()
        self.get_results()

    def run_tds(self):
        """
        Performs the numerical integration using the chosen method.
        """
        while self.t < self.t_final:
            converged = self.integrator.step(
                dae=self.system.dae,
                dt=self.dt,
                method=self.integrator,
                tol=config.TOL,
                max_iter=config.MAX_ITER
            )

            if not converged:
                raise RuntimeError(f"Integration step did not converge at time {self.t}.")

            self.t += self.dt
            self.results.append((self.t, self.system.dae.x.copy(), self.system.dae.y.copy()))

            self.get_events()

    def run_steadystate(self):
        """
        Performs steady-state computation.
        """
        converged = self.steadystate.steadystate(dae=self.system.dae, method=self.steadystate, tol=config.TOL, max_iter=config.MAX_ITER)
        
        if converged:
                print(f"Steady-state found.")
        else:
            raise RuntimeError("Steady-state not found.")

    def get_results(self):
        self.data_processor.save_data(self.results)
        # self.data_processor.export_csv()
        self.data_processor.plot_results()
        # self.data_processor.compare_with_andes()

    def get_events(self):
        for i, event in enumerate(self.events):
            if i in self.applied_events:
                continue  # already applied

            if abs(self.t - event["time"]) < 1e-6:
                model = self.system.devices[event["model"]]
                param = getattr(model, event["param"])

                if isinstance(param, NumDynParam):
                    print(f"Applying event at t={self.t:.2f}: "
                        f"{event['model']}[{event['device_index']}].{event['param']} = {event['value']}")
                    param.value[event["device_index"]] = event["value"]
                    self.applied_events.append(i)
