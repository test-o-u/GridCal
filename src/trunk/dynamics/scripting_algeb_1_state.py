# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import math

import numpy as np
from matplotlib import pyplot as plt
from GridCalEngine.Utils.Symbolic.symbolic import Const, Var, cos, sin
from GridCalEngine.Utils.Symbolic.block import Block
from GridCalEngine.Utils.Symbolic.block_solver import BlockSolver

# ----------------------------------------------------------------------------------------------------------------------
# Line
# ----------------------------------------------------------------------------------------------------------------------
g = Const(1.1)
b = Const(-12.1)
bsh = Const(0.0000001)
Qline_from = Var("Qline_from")
Qline_to = Var("Qline_to")
Pline_from = Var("Pline_from")
Pline_to = Var("Pline_to")
Vline_from = Var("Vline_from")
Vline_to = Var("Vline_to")
dline_from = Var("dline_from")
dline_to = Var("dline_to")

line_block = Block(
    algebraic_eqs=[
        Pline_from - ((Vline_from ** 2 * g) - g * Vline_from * Vline_to * cos(dline_from - dline_to) + b * Vline_from * Vline_to * cos(dline_from - dline_to + np.pi/2)),
        Qline_from - (Vline_from ** 2 * (-bsh/2 - b) - g * Vline_from * Vline_to * sin(dline_from - dline_to) + b * Vline_from * Vline_to * sin(dline_from - dline_to + np.pi/2)),
        Pline_to - ((Vline_to ** 2 * g) - g * Vline_to * Vline_from * cos(dline_to - dline_from) + b * Vline_to * Vline_from * cos(dline_to - dline_from + np.pi/2)),
        Qline_to - (Vline_to ** 2 * (-bsh/2 - b) - g * Vline_to * Vline_from * sin(dline_to - dline_from) + b * Vline_to * Vline_from * sin(dline_to - dline_from + np.pi/2)),
    ],
    algebraic_vars=[dline_from, Vline_from, dline_to, Vline_to],
)

# ----------------------------------------------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------------------------------------------
coeff_alfa = Const(1.8)
Pl0 = Const(0.1)
Ql0 = Const(0.2)
coeff_beta = Const(8.0)
Ql = Var("Ql")
Pl = Var("Pl")

load_block = Block(
    algebraic_eqs=[
        # Pl - (Pl0 * Ql ** coeff_alfa),
        # Ql - (Ql0 * Ql ** coeff_beta)
        Pl - (Pl0),
        Ql - (Ql0)
    ],
    algebraic_vars=[Pl, Ql]
)

# ----------------------------------------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------------------------------------
pi = Const(math.pi)
fn = Const(50.0)
tm = Const(10.0)
M = Const(1.0)
D = Const(0.003)
ra = Const(0.3)
xd = Const(0.86138701)
vf = Const(3.81099313)

# delta = Var("delta")
# omega = Var("omega")
# psid = Var("psid")
# psiq = Var("psiq")
# i_d = Var("i_d")
# i_q = Var("i_q")
# v_d = Var("v_d")
# v_q = Var("v_q")
# t_e = Var("t_e")
# P_e = Var("P_e")
# Q_e = Var("Q_e")
Pg = Var("Pg")
Qg = Var("Qg")

# generator_block = Block(
#     state_eqs=[
#         # delta - (2 * pi * fn) * (omega - 1),
#         # omega - (-tm / M + t_e / M - D / M * (omega - 1))
#         (2 * pi * fn) * (omega - 1),  # dδ/dt
#         (-tm + t_e - D * (omega - 1)) / M  # dω/dt
#     ],
#     state_vars=[delta, omega],
#     algebraic_eqs=[
#         # psid - ((-ra*i_q + v_q) - psid),
#         psid - (-ra * i_q + v_q),
#         # psiq - ((-ra * i_d + v_d) - psiq),
#         psiq - (-ra * i_d + v_d),
#         i_d - (psid + xd * i_d - vf),
#         i_q - (psiq + xd * i_q),
#         v_d - (Qg * sin(delta - Pg)),
#         v_q - (Qg * cos(delta - Pg)),
#         t_e - ((psid * i_q - psiq * i_d)),
#         # P_e - (v_d * i_d + v_q * i_q - P_e),
#         # Q_e - (v_q * i_d - v_d * i_q - Q_e),
#         P_e - (v_d * i_d + v_q * i_q),
#         Q_e - (v_q * i_d - v_d * i_q),
#         Pg - (v_d * i_d + v_q * i_q),
#         Qg - (v_q * i_d + v_d * i_q)
#     ],
#     algebraic_vars=[psid, psiq, i_d, i_q, v_d, v_q, t_e, P_e, Q_e, Pg, Qg]
# )

# generator_block = Block(
#     algebraic_eqs=[
#         Pg - Pline_from,
#         Qg - Qline_from,
#     ],
#     algebraic_vars=[Pg, Qg]
# )

foo = Var("foo")
dummy_generator_block = Block(
    state_eqs=[
        -10 * foo,  # dfoo/dt = -10 * foo, so should be stable at 0
    ],
    state_vars=[foo]
)

# ----------------------------------------------------------------------------------------------------------------------
# Bus
# ----------------------------------------------------------------------------------------------------------------------
bus1_block = Block(
    algebraic_eqs=[
        Vline_from - 1.0,
        dline_from - 0.0,
    ],
    algebraic_vars=[Pline_from, Qline_from],
)

bus2_block = Block(
    algebraic_eqs=[
        Pl + Pline_to,
        Ql + Qline_to,       
    ],
    algebraic_vars=[Pline_to, Qline_to],
)

# ----------------------------------------------------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------------------------------------------------

sys = Block(
    children=[line_block, load_block, bus1_block, bus2_block, dummy_generator_block],
    in_vars=[],
    out_vars=[]
)

# ----------------------------------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------------------------------
slv = BlockSolver(sys)

mapping = {
    # delta: 0.0,
    # omega: 1.0,
    # P1: np.deg2rad(15),  # rotor angle (rad)
    # Q1: 1.0,  # generator terminal voltage magnitude (pu)

    # P2: np.deg2rad(10),  # angle of second bus
    # Q2: 0.95,  # remote bus voltage (pu)

    foo: 0.5,

    dline_from: 0.0,
    dline_to: 0.0,
    Vline_from: 1.0,
    Vline_to: 1.0,

    # Pg: np.deg2rad(15),  # P1
    # Qg: 0.23,  # Q1

    Pl: 0.1,  # P2
    Ql: 0.2,  # Q2

    # Pline_from: np.deg2rad(15),  # P1
    # Qline_from: 1.0,  # Q1
    # Pline_to: np.deg2rad(10),  # P2
    # Qline_to: 0.95,  # Q2

    Pline_from: 0.1,
    Qline_from: 0.2,
    Pline_to: -0.1,
    Qline_to: -0.2,

    # psid: 1.0,  # d-axis flux linkage (pu)
    # psiq: 0.0,  # q-axis flux linkage (pu)
    # i_d: 0.1,  # d-axis stator current (pu)
    # i_q: 0.2,  # q-axis stator current (pu)
    # v_d: 0.0,  # d-axis voltage (pu)
    # v_q: 1.0,  # q-axis voltage (pu)
    # t_e: 0.1,  # electromagnetic torque (pu)
    # P_e: 0.1,  # real power (pu)
    # Q_e: 0.2,  # reactive power (pu)
}

x0 = slv.build_init_vector(mapping)
vars_in_order = slv.sort_vars(mapping)

t, y = slv.simulate(
    t0=0,
    t_end=2.0,
    h=0.001,
    x0=x0,
    method="implicit_euler"
)

fig = plt.figure(figsize=(12, 8))
# plt.plot(t, y)
# plt.plot(t, y[:, slv.get_var_idx(omega)], label="ω (pu)")
# plt.plot(t, y[:, slv.get_var_idx(delta)], label="δ (rad)")
plt.plot(t, y[:, slv.get_var_idx(Vline_from)], label="Vline_from (pu)")
plt.plot(t, y[:, slv.get_var_idx(Vline_to)], label="Vline_to (pu)")
# plt.plot(t, y[:, slv.get_var_idx(Pline_from)], label="Pline_from (pu)")
# plt.plot(t, y[:, slv.get_var_idx(Qline_from)], label="Qline_from (pu)")
# plt.plot(t, y[:, slv.get_var_idx(Pline_to)], label="Pline_to (pu)")
# plt.plot(t, y[:, slv.get_var_idx(Qline_to)], label="Qline_to (pu)")
plt.plot(t, y[:, slv.get_var_idx(foo)], label="foo (pu)")
plt.legend()
plt.show()
