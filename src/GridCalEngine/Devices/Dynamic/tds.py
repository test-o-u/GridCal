import pdb
from GridCalEngine.Devices.Dynamic.integration import method_map


class TDS():
    """
    Time domain simulation class.
    """
    def __init__(self, system, dt=0.1, t_final=1.0, method="trapezoid"):
        self.system = system
        self.dt = dt
        self.t_final = t_final
        self.method = method
        self.results = []

        # Get integration method
        if method not in method_map:
            raise ValueError(f"Unknown integration method: {method}")
        self.integrator = method_map[method]

        # Run simulation
        self.system.dae.initilize_fg()
        # self.run()

    def run(self):
        """
        Performs the numerical integration using the chosen method.
        """
        t = 0
        while t < self.t_final:
            # Solve DAE step
            converged = self.integrator.step(dae=self.system.dae, dt=self.dt, method=self.integrator)

            if not converged:
                raise RuntimeError("Integration step did not converge")
            
            t += self.dt
            self.results.append((t, self.system.dae.x.copy(), self.system.dae.y.copy()))
