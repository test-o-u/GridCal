# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import GridCalEngine.Devices.Dynamic.io.config as config
from GridCalEngine.Devices.Dynamic.integration import method_map

class TDS():
    """
    Time domain simulation class.
    """
    def __init__(self, system):

        self.system = system

        self.dt = config.TIME_STEP
        self.t_final = config.SIMULATION_TIME
        self.method_tds = config.INTEGRATION_METHOD
        self.method_ss = config.STEADYSTATE_METHOD

        self.results = []

        # Get integration method
        if self.method_tds not in method_map or self.method_ss not in method_map:
            raise ValueError(f"Unknown integration method: {self.method_tds}")
        self.integrator = method_map[self.method_tds]
        self.steadystate = method_map[self.method_ss]
 
        # Initialize 
        self.system.dae.initilize_fg()

        # Run simulation
        self.run_tds()

    def run_tds(self):
        """
        Performs the numerical integration using the chosen method.
        """
        t = 0
        while t < self.t_final:
            # Solve DAE step
            converged = self.integrator.step(dae=self.system.dae, dt=self.dt, method=self.integrator, tol=config.TOL, max_iter=config.MAX_ITER)

            if not converged:
                raise RuntimeError("Integration step did not converge")
            
            t += self.dt
            self.results.append((t, self.system.dae.x.copy(), self.system.dae.y.copy()))

    def run_steadystate(self):
        """
        Performs steady-state computation.
        """

        converged = self.steadystate.steadystate(dae=self.system.dae, method=self.steadystate, tol=config.TOL, max_iter=config.MAX_ITER)
        
        if converged:
                print(f"Steady-state found.")
        else:
            raise RuntimeError("Steady-state not found")