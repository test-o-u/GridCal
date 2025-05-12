# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pdb
import matplotlib.pyplot as plt

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

        self.tds_plot = config.TDS_PLOT

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
        # self.get_results()

    def run_tds(self):
        """
        Performs the numerical integration using the chosen method.
        """
        if self.tds_plot:
            fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

            # Setup x[1] plot
            line_x, = axs[0].plot([], [], label='x[1]')
            axs[0].set_ylabel(r"$\omega$")
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_ylim(0.9, 1.1)  # fixed y-axis range
            axs[0].set_xlim(0, self.t_final) 

            # Setup y[0] plot
            line_y, = axs[1].plot([], [], label='y[0]', color='orange')
            axs[1].set_ylabel(r'$P_{Gen}$')	
            axs[1].legend()
            axs[1].grid(True)
            axs[1].set_ylim(-0.1, 2.0)
            axs[1].set_xlim(0, self.t_final) 

            # Setup y[2] plot
            line_param, = axs[2].plot([], [], label='y[2]', color='green')
            axs[2].set_ylabel(r'$P_{Load}$')
            axs[2].set_xlabel(r'Time [s]')
            axs[2].legend()
            axs[2].grid(True)
            axs[2].set_ylim(-0.1, 2.0)
            axs[2].set_xlim(0, self.t_final)  

            plt.ion()
            plt.tight_layout()
            plt.show()

            time_data, x_data = [], []
            y_data, param_data = [], []

        try:
            while self.t < self.t_final:
                converged = self.integrator.step(
                    dae=self.system.dae,
                    dt=self.dt,
                    method=self.integrator,
                    tol=config.TOL,
                    max_iter=config.MAX_ITER,
                    step_plot=config.STEP_PLOT
                )

                if not converged:
                    if self.tds_plot:
                        plt.ioff()
                        plt.show()
                    raise RuntimeError(f"Integration step did not converge at time {self.t}.")

                if self.tds_plot:
                    time_data.append(self.t)
                    x_data.append(self.system.dae.x[1])
                    y_data.append(self.system.dae.y[11])
                    param_data.append(self.system.models['ExpLoad'].Pl0.value[0])

                    # Update plots
                    line_x.set_data(time_data, x_data)
                    line_y.set_data(time_data, y_data)
                    line_param.set_data(time_data, param_data)

                    plt.pause(0.01)

                self.t += self.dt
                self.results.append((self.t, self.system.dae.x.copy(), self.system.dae.y.copy()))
                
                print(f"=====Time step: {self.t}==========\n")
                print(f"Pe: {self.system.dae.y.copy()[11]}\n")
                print(f"te: {self.system.dae.y.copy()[10]}\n")
                print(f"tm: {self.system.dae.y.copy()[13]}\n")
                print(f"==================================\n")

                self.get_events()

        finally:
            if self.tds_plot:
                plt.ioff()
                plt.show()


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
