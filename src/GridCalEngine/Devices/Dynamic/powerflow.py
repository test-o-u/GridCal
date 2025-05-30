# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os
import pandas as pd
import numpy as np
#from pandas.conftest import float_numpy_dtype
from scipy.optimize import newton_krylov, NoConvergence
import sympy as sp


from GridCalEngine.IO.file_handler import FileOpen
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions, multi_island_pf_nc
from GridCalEngine.Simulations.PowerFlow.power_flow_options import SolverType
from GridCalEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from GridCalEngine.Compilers.circuit_to_data import compile_numerical_circuit_at
import GridCalEngine.api as gce

# Using Gridcal
#
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# # 'IEEE 14 bus.raw', 'IEEE 14 bus.sav.xlsx'
# # 'IEEE 30 bus.raw', 'IEEE 30 bus.sav.xlsx'
# # 'IEEE 118 Bus v2.raw', 'IEEE 118 Bus.sav.xlsx'
# #'IEEE 14 bus.raw'
#
#
# file = 'IEEE 14 bus.raw'
#
# # SolverType.IWAMOTO
# # SolverType.LM
# # SolverType.FASTDECOUPLED
# # SolverType.PowellDogLeg
# # SolverType.HELM
#
# solver_type = SolverType.NR
#
# options = PowerFlowOptions(solver_type,
#                                    verbose=0,
#                                    control_q=False,
#                                    retry_with_other_methods=False)
#
#
# #fname = os.path.join('src','GridCalEngine', 'Devices', 'Dynamic', 'io', 'data', file)
# main_circuit = FileOpen(file).open()
# power_flow = PowerFlowDriver(main_circuit, options)
# power_flow.run()
# results = power_flow.results
# df_bus, df_branch = results.export_all()
#
# print(df_bus)
# print(df_branch)


# Using Newton_Krylov

class PowerFlow():

    """
    Powerflow class.
    """



    def __init__(self):
        """
        PF constructor
        Initializes and executes time domain simulation.
        :param system: The system instance containing models and devices
        """

        #variables


        self.delta = 0
        self.omega = 1
        self.P_origin = 2
        self.P_end = 3
        self.Q_origin = 4
        self.Q_end = 5
        self.psid = 6
        self.psiq = 7
        self.i_d = 8
        self.i_q = 9
        self.v_d = 10
        self.v_q = 11
        self.t_e = 12
        self.P_e = 13
        self.Q_e = 14


        self.g =  0.09
        self.b = -20.99
        self.bsh =  -0.000001

        self.fn =  50.0
        self.D = 10
        self.M = 1.0
        self.ra = 0.003
        self.xd = 0.3
        self.tm = 0.86138701
        self.vf = 3.81099313


        self.coeff_alfa = 0.0
        self.coeff_beta = 0.0
        self.Pl0 = 0.099
        self.Ql0 = 0.198


        #parameters




    def fun(self, x):
        return [(2 * np.pi * self.fn) * (x[self.omega] - 1),
                (-self.tm / self.M + x[self.t_e] / self.M - self.D / self.M * (x[self.omega] - 1)),
                (-self.ra * x[self.i_q] + x[self.v_q]) - x[self.psid],
                (-self.ra * x[self.i_d] + x[self.v_d]) - x[self.psiq],
                x[self.psid] + self.xd * x[self.i_d] - self.vf,
                x[self.psiq] + self.xd * x[self.i_q],
                x[self.Q_origin] * np.sin(x[self.delta] - x[self.P_origin]) - x[self.v_d],
                x[self.Q_origin] * np.cos(x[self.delta] - x[self.P_origin]) - x[self.v_q],
                (x[self.psid] * x[self.i_q] - x[self.psiq] * x[self.i_d]) - x[self.t_e],
                (x[self.v_d] * x[self.i_d] + x[self.v_q] * x[self.i_q]) - x[self.P_e],
                (x[self.v_q] * x[self.i_d] - x[self.v_d] * x[self.i_q]) - x[self.Q_e],
                (x[self.Q_origin] ** 2 * self.g - x[self.Q_origin] * x[self.Q_end] * (self.g * np.cos(x[self.P_origin] - x[self.P_end]) + self.b * np.sin(x[self.P_origin] - x[self.P_end]))) + (x[self.v_d] * x[self.i_d] + x[self.v_q] * x[self.i_q]),
                (- x[self.Q_origin] ** 2 * (self.b + self.bsh / 2) - x[self.Q_origin] * x[self.Q_end] * (self.g * np.sin(x[self.P_origin] - x[self.P_end]) - self.b * np.cos(x[self.P_origin] - x[self.P_end]))) + (x[self.v_q] * x[self.i_d] - x[self.v_d] * x[self.i_q]),
                x[self.Q_end] ** 2 * self.g - x[self.Q_end] * x[self.Q_origin] * (self.g * np.cos(x[self.P_end] - x[self.P_origin]) + self.b * np.sin(x[self.P_end] - x[self.P_origin])) + self.Pl0 * x[self.Q_end] ** self.coeff_alfa,
                (- x[self.Q_end] ** 2 * (self.b + self.bsh / 2) - x[self.Q_end] * x[self.Q_origin] * (self.g * np.sin(x[self.P_end] - x[self.P_origin]) - self.b * np.cos(x[self.P_end] - x[self.P_origin]))) + self.Ql0 * x[self.Q_origin] ** self.coeff_beta]



    def calc_powerflow(self):
        try:
            sol = newton_krylov(self.fun, [0.0, 1.0, 0, 0, 1.0, 0.95, 1.0, 0.0, 0.1, 0.2, 0.0, 1.0, 0.1, 0.2, 0.2])
        except NoConvergence as e:
            print('no convergence')
            sol = e.args[0]  # This contains the last iterate
        print(sol)
        return sol

my_power_flow = PowerFlow()
my_power_flow.calc_powerflow()

# 'delta': 0.0,
# 'omega': 1.0,
# 'a1': 15 * (np.pi / 180),  # rotor angle (rad)
# 'a2': 10 * (np.pi / 180),  # angle of second bus
# 'v1': 1.0,                 # generator terminal voltage magnitude (pu)
# 'v2': 0.95,                # remote bus voltage (pu)
# 'psid': 1.0,  # d-axis flux linkage (pu)
# 'psiq': 0.0,  # q-axis flux linkage (pu)
# 'i_d': 0.1,  # d-axis stator current (pu)
# 'i_q': 0.2,  # q-axis stator current (pu)
# 'vd': 0.0,  # d-axis voltage (pu)
# 'vq': 1.0,  # q-axis voltage (pu)
# 'te': 0.1,  # electromagnetic torque (pu)
# 'Pe': 0.2,  # real power (pu)
# 'Qe': 0.2,  # reactive power (pu)




