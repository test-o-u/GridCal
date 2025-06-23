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


# Line
g = Const(0.5)
b = Const(1.2)
bsh = Const(0.3)

# Load
coeff_alfa = Const(1.8)
Pl0 = Const(10.0)
Ql0 = Const(9.0)
coeff_beta = Const(8.0)

# Generator
pi = Const(math.pi)
# fn = Var("fn")
# tm = Var("tm")
# M = Var("M")
# D = Var("D")
# ra = Var("ra")
# xd = Var("xd")
# vf = Var("vf")
fn = Const(50.0)
tm = Const(10.0)
M = Const(1.0)
D = Const(0.003)
ra = Const(0.3)
xd = Const(0.86138701)
vf = Const(3.81099313)
# Define variables

# Line
Qline_from = Var("Q_origin")
Qline_to = Var("Q_end")
Pline_from = Var("P_origin")
Pline_to = Var("P_end")

# Load
Ql = Var("Ql")
Pl = Var("Pl")

# ----------------------------------------------------------------------------------------------------------------------
# Line
# ----------------------------------------------------------------------------------------------------------------------

line_block = Block(
    algebraic_eqs=[
        (Qline_from ** 2 * g) - (
                Qline_from * Qline_to * (g * cos(Pline_from - Pline_to) + b * sin(Pline_from - Pline_to))),
        -Qline_from ** 2 * (b + bsh / Const(2)) - Qline_from * Qline_to * (
                g * sin(Pline_from - Pline_to) - b * cos(Pline_from - Pline_to)),
        (Qline_to ** 2 * g) - (
                Qline_to * Qline_from * (g * cos(Pline_to - Pline_from) + b * sin(Pline_to - Pline_from))),
        -Qline_to ** 2 * (b + bsh / Const(2)) - Qline_to * Qline_from * (
                g * sin(Pline_to - Pline_from) - b * cos(Pline_to - Pline_from))
    ],
    algebraic_vars=[Pline_from, Qline_from, Pline_to, Qline_to],
)

# ----------------------------------------------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------------------------------------------


load_block = Block(
    algebraic_eqs=[
        Pl0 * Ql ** coeff_alfa,
        Ql0 * Ql ** coeff_beta
    ],
    algebraic_vars=[Ql, Pl]
)

# ----------------------------------------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------------------------------------
# Generator
delta = Var("delta")
omega = Var("omega")
psid = Var("psid")
psiq = Var("psiq")
i_d = Var("i_d")
i_q = Var("i_q")
v_d = Var("v_d")
v_q = Var("v_q")
t_e = Var("t_e")
P_e = Var("P_e")
Q_e = Var("Q_e")
Pg = Var("Pg")
Qg = Var("Qg")

generator_block = Block(
    state_eqs=[
        delta - (2 * pi * fn) * (omega - 1),
        omega - (-tm / M + t_e / M - D / M * (omega - 1))
    ],
    state_vars=[delta, omega],
    algebraic_eqs=[
        psid - ((-ra * i_q + v_q) - psid),
        psiq - ((-ra * i_d + v_d) - psiq),
        i_d - (psid + xd * i_d - vf),
        i_q - (psiq + xd * i_q),
        v_d - (Qg * sin(delta - Pg)),
        v_q - (Qg * cos(delta - Pg)),
        t_e - ((psid * i_q - psiq * i_d) - t_e),
        P_e - (v_d * i_d + v_q * i_q - P_e),
        Q_e - (v_q * i_d - v_d * i_q - Q_e),
        Pg - (v_d * i_d + v_q * i_q),
        Qg - (v_q * i_d + v_d * i_q)
    ],
    algebraic_vars=[psid, psiq, i_d, i_q, v_d, v_q, t_e, P_e, Q_e, Pg, Qg]
)

# ----------------------------------------------------------------------------------------------------------------------
# Buses
# ----------------------------------------------------------------------------------------------------------------------
P1 = Var("P1")
Q1 = Var("Q1")
P2 = Var("P2")
Q2 = Var("Q2")

bus1_block = Block(
    algebraic_eqs=[
        P1 - (Pg - Pline_from),
        Q1 - (Qg - Qline_from)
    ],
    algebraic_vars=[P1, Q1]
)

bus2_block = Block(
    algebraic_eqs=[
        P2 - (-Pl + Pline_to),
        Q2 - (-Ql + Qline_to)
    ],
    algebraic_vars=[P2, Q2]
)

# ----------------------------------------------------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------------------------------------------------

sys = Block(
    children=[line_block, load_block, generator_block, bus1_block, bus2_block],
    in_vars=[],
    out_vars=[]
)

# ----------------------------------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------------------------------

slv = BlockSolver(sys)
t, y = slv.simulate(
    t0=0,
    t_end=3.0,
    h=0.01,
    x0=slv.build_init_vector({
        delta: 0.0,
        omega: 1.0,
        P1: np.deg2rad(15),  # rotor angle (rad)
        Q1: 1.0,  # generator terminal voltage magnitude (pu)

        P2: np.deg2rad(10),  # angle of second bus
        Q2: 0.95,  # remote bus voltage (pu)

        Pg: np.deg2rad(15),  # P1
        Qg: 1.0,  # Q1

        Pl: np.deg2rad(10),  # P2
        Ql: 0.95,  # Q2

        Pline_from: np.deg2rad(15),  # P1
        Qline_from: 1.0,  # Q1

        Pline_to: np.deg2rad(10),  # P2
        Qline_to: 0.95,  # Q2

        psid: 1.0,  # d-axis flux linkage (pu)
        psiq: 0.0,  # q-axis flux linkage (pu)
        i_d: 0.1,  # d-axis stator current (pu)
        i_q: 0.2,  # q-axis stator current (pu)
        v_d: 0.0,  # d-axis voltage (pu)
        v_q: 1.0,  # q-axis voltage (pu)
        t_e: 0.1,  # electromagnetic torque (pu)
        P_e: 0.2,  # real power (pu)
        Q_e: 0.2,  # reactive power (pu)
    })
)

fig = plt.figure(figsize=(12, 8))
plt.plot(t, y)