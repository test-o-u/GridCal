import pdb
from GridCalEngine.Devices.Dynamic.integration import method_map


class TDS():
    """
    Time domain simulation class.
    """
    def __init__(self, system, dt=0.0001, t_final=0.1, method_ss="steadystate", method_tds="trapezoid"):
        self.system = system
        self.dt = dt
        self.t_final = t_final
        self.method_tds = method_tds
        self.method_ss = method_ss
        self.results = []

        # Get integration method
        if self.method_tds not in method_map or self.method_ss not in method_map:
            raise ValueError(f"Unknown integration method: {self.method_tds}")
        self.integrator = method_map[self.method_tds]
        self.steadystate = method_map[self.method_ss]
 
        # Initialize 
        self.system.dae.initilize_fg()
        # Compute steady-state
        self.run_steadystate()
        # Run simulation
        # self.run_tds()

    def run_tds(self):
        """
        Performs the numerical integration using the chosen method.
        """
        t = 0
        while t < self.t_final:
            # Solve DAE step
            converged = self.integrator.step(dae=self.system.dae, dt=self.dt, method=self.integrator)

            if converged:
                print(f"Converged at time {t}")
            else:
                raise RuntimeError("Integration step did not converge")
            
            t += self.dt
            self.results.append((t, self.system.dae.x.copy(), self.system.dae.y.copy()))

    def run_steadystate(self):
        """
        Performs steady-state computation.
        """

        converged = self.steadystate.steadystate(dae=self.system.dae, method=self.steadystate)
        
        if converged:
                print(f"Steady-state found.")
        else:
            raise RuntimeError("Steady-state not found")