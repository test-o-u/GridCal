# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import math

import numpy as np
from matplotlib import pyplot as plt
from pygments.lexers.dsls import VGLLexer

from GridCalEngine.Utils.Symbolic.symbolic import Const, Var, cos, sin
from GridCalEngine.Utils.Symbolic.block import Block
from GridCalEngine.Utils.Symbolic.block_solver import BlockSolver

# ----------------------------------------------------------------------------------------------------------------------
# Line
# ----------------------------------------------------------------------------------------------------------------------
g = Const(0.5)
b = Const(1.2)
bsh = Const(0.3)
Qline_from = Var("Qline_from")
Qline_to = Var("Qline_to")
Pline_from = Var("Pline_from")
Pline_to = Var("Pline_to")
Vline_from = Var("Vline_from")
Vline_to = Var("Vline_to")
dline_from = Var("dline_from")
dline_to = Var("dline_to")

Ql = Var("Ql")
Pl = Var("Pl")
Vl = Var("Vl")

delta = Var("delta")
omega = Var("omega")
psid = Var("psid")
psiq = Var("psiq")
i_d = Var("i_d")
i_q = Var("i_q")
v_d = Var("v_d")
v_q = Var("v_q")
t_e = Var("t_e")
p_g = Var("P_e")
Q_g = Var("Q_e")
Vg = Var("Vg")
dg = Var("dg")

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
Pl0 = Const(10.0)
Ql0 = Const(9.0)
coeff_beta = Const(8.0)


load_block = Block(
    algebraic_eqs=[
        Pl - (Pl0 * Vl ** coeff_alfa),
        Ql - (Ql0 * Vl ** coeff_beta)
    ],
    algebraic_vars=[Ql, Pl]
)

# ----------------------------------------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------------------------------------
# Generator
pi = Const(math.pi)
fn = Const(50.0)
tm = Const(10.0)
M = Const(1.0)
D = Const(0.003)
ra = Const(0.3)
xd = Const(0.86138701)
vf = Const(3.81099313)




generator_block = Block(
    state_eqs=[
        # delta - (2 * pi * fn) * (omega - 1),
        # omega - (-tm / M + t_e / M - D / M * (omega - 1))
        (2 * pi * fn) * (omega - 1),  # dδ/dt
        (-tm + t_e - D * (omega - 1)) / M  # dω/dt
    ],
    state_vars=[delta, omega],
    algebraic_eqs=[
        psid - (-ra * i_q + v_q),
        psiq - (-ra * i_d + v_d),
        i_d - (psid + xd * i_d - vf),
        i_q - (psiq + xd * i_q),
        v_d - (Vg * sin(delta - dg)),
        v_q - (Vg * cos(delta - dg)),
        t_e - (psid * i_q - psiq * i_d),
        (v_d * i_d + v_q * i_q) - p_g,
        (v_q * i_d - v_d * i_q) - Q_g
    ],
    algebraic_vars=[psid, psiq, i_d, i_q, v_d, v_q, t_e, p_g, Q_g]
)

# ----------------------------------------------------------------------------------------------------------------------
# Buses
# ----------------------------------------------------------------------------------------------------------------------
# P1 = Var("P1")
# Q1 = Var("Q1")
# P2 = Var("P2")
# Q2 = Var("Q2")

bus1_block = Block(
    algebraic_eqs=[
        p_g - Pline_from,
        Q_g - Qline_from,
        Vg - Vline_from,
        dg - dline_from
    ],
    algebraic_vars=[Pline_from, Qline_from, Vg, dg]
)

bus2_block = Block(
    algebraic_eqs=[
        Pl + Pline_to,
        Ql + Qline_to,
        Vl - Vline_to
    ],
    algebraic_vars=[Pline_to, Qline_to, Vl]
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

mapping = {

    dline_from: 15 * (np.pi / 180),
    dline_to: 10 * (np.pi / 180),
    Vline_from: 1.0,
    Vline_to: 0.95,
    Vl: 0.95,
    Vg: 1.0,
    dg: 15 * (np.pi / 180),

    Pline_from: 0.1,
    Qline_from: 0.2,
    Pline_to: 0.1,
    Qline_to: 0.2,


    Pl: -0.1,  # P2
    Ql: -0.2,  # Q2


    delta: 0.0,
    omega: 1.0,
    psid: 3.825,  # d-axis flux linkage (pu)
    psiq: 0.0277,  # q-axis flux linkage (pu)
    i_d: 0.1,  # d-axis stator current (pu)
    i_q: 0.2,  # q-axis stator current (pu)
    v_d: -0.2588,  # d-axis voltage (pu)
    v_q:  0.9659,  # q-axis voltage (pu)
    t_e: 0.1,  # electromagnetic torque (pu)
    p_g: 0.1673,
    Q_g: 0.1484
}

x0 = slv.build_init_vector(mapping)
vars_in_order = slv.sort_vars(mapping)

t, y = slv.simulate(
    t0=0,
    t_end=1.0,
    h=0.001,
    x0=x0,
    method="implicit_euler"
)

fig = plt.figure(figsize=(12, 8))
# plt.plot(t, y)
plt.plot(t, y[:, slv.get_var_idx(omega)], label="ω (pu)")
plt.plot(t, y[:, slv.get_var_idx(delta)], label="δ (rad)")
# plt.plot(t, y[:, slv.get_var_idx(Vline_from)], label="Vline_from (Vlf)")
# plt.plot(t, y[:, slv.get_var_idx(Vline_to)], label="Vline_to (Vlt)")
plt.legend()
plt.show()
