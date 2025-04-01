# from GridCalEngine.Devices.Dynamic import Integration

class TDS():

    def __init__(self, system, dt=0.01, t_final=10.0, method="backward_euler"):
        self.system = system
        self.dt = dt
        self.t_final = t_final
        self.method = method
        self.results = []  # Stores simulation results

    def run(self):
        """
        Performs the numerical integration using the chosen method.
        """
        t = 0
        while t < self.t_final:
            # Solve DAE at current time step
            self.step()
            t += self.dt
            self.results.append((t, self.system.dae.x.copy(), self.system.dae.y.copy()))
