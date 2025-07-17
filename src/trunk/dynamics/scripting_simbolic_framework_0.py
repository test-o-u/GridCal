# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import math
import pdb

import numpy as np
from matplotlib import pyplot as plt

from GridCalEngine.Simulations.Rms.initialization import initialize
#from pygments.lexers.dsls import VGLLexer

#from GridCalEngine.Utils.Symbolic.events import EventParam
from GridCalEngine.Utils.Symbolic.symbolic import Const, Var, cos, sin, EventParam
from GridCalEngine.Utils.Symbolic.block import Block
from GridCalEngine.Utils.Symbolic.block_solver import BlockSolver
import GridCalEngine.api as gce

# ----------------------------------------------------------------------------------------------------------------------
# Line
# ----------------------------------------------------------------------------------------------------------------------
g = Const(5)
b = Const(-12)
bsh = Const(0.03)
Qline_from = Var("Qline_from")
Qline_to = Var("Qline_to")
Pline_from = Var("Pline_from")
Pline_to = Var("Pline_to")
Vline_from = Var("Vline_from")
Vline_to = Var("Vline_to")
dline_from = Var("dline_from")
dline_to = Var("dline_to")



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
tm = Var("tm")
et = Var("et")

line_block = Block(
    algebraic_eqs=[
        Pline_from - ((Vline_from ** 2 * g) - g * Vline_from * Vline_to * cos(dline_from - dline_to) + b * Vline_from * Vline_to * cos(dline_from - dline_to + np.pi/2)),
        Qline_from - (Vline_from ** 2 * (-bsh/2 - b) - g * Vline_from * Vline_to * sin(dline_from - dline_to) + b * Vline_from * Vline_to * sin(dline_from - dline_to + np.pi/2)),
        Pline_to - ((Vline_to ** 2 * g) - g * Vline_to * Vline_from * cos(dline_to - dline_from) + b * Vline_to * Vline_from * cos(dline_to - dline_from + np.pi/2)),
        Qline_to - (Vline_to ** 2 * (-bsh/2 - b) - g * Vline_to * Vline_from * sin(dline_to - dline_from) + b * Vline_to * Vline_from * sin(dline_to - dline_from + np.pi/2)),
    ],
    algebraic_vars=[dline_from, Vline_from, dline_to, Vline_to],
    parameters=[g, b, bsh],
    events=[]
)

# ----------------------------------------------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------------------------------------------
coeff_alfa = Const(1.8)
Pl0 = EventParam(0.1, 2, 50, 'Pl0')
Ql0 = EventParam(0.1, 2, 50, 'Ql0')
coeff_beta = Const(8.0)

Ql = Var("Ql")
Pl = Var("Pl")

load_block = Block(
    algebraic_eqs=[
        Pl - (Pl0),
        Ql - (Ql0)
    ],
    algebraic_vars=[Ql, Pl],
    parameters=[],
    events=[Pl0, Ql0]
)

# ----------------------------------------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------------------------------------
# Generator
pi = Const(math.pi)
fn =  Const(50)
# tm = Const(0.1)
M = Const(1.0)
D = EventParam(100, 100, 50, 'D')
#D = Const(100)
ra = Const(0.3)
xd = Const(0.86138701)
vf = Const(0.9584725405467506) #1.081099313

Kp = Const(1.0)
Ki = Const(10.0)
Kw = Const(10.0)



generator_block = Block(
    state_eqs=[
        # delta - (2 * pi * fn) * (omega - 1),
        # omega - (-tm / M + t_e / M - D / M * (omega - 1))
        (2 * pi * fn) * (omega - 1),  # dδ/dt
        (tm - t_e - D * (omega - 1)) / M,  # dω/dt
        -Kp * et - Ki * et - Kw * (omega - 1)  # det/dt
    ],
    state_vars=[delta, omega, et],
    algebraic_eqs=[
        et - (tm - t_e),
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
    algebraic_vars=[tm, psid, psiq, i_d, i_q, v_d, v_q, t_e, p_g, Q_g],
    parameters=[fn, M, ra, xd, vf, Kp, Ki, Kw],
    events=[D]
)

# ----------------------------------------------------------------------------------------------------------------------
# Buses
# ----------------------------------------------------------------------------------------------------------------------

bus1_block = Block(
    algebraic_eqs=[
        p_g - Pline_from,
        Q_g - Qline_from,
        Vg - Vline_from,
        dg - dline_from
    ],
    algebraic_vars=[Pline_from, Qline_from, Vg, dg],
    events=[]
)

bus2_block = Block(
    algebraic_eqs=[
        Pl + Pline_to,
        Ql + Qline_to,
    ],
    algebraic_vars=[Pline_to, Qline_to],
    events=[]
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
# Intialization
# ----------------------------------------------------------------------------------------------------------------------
grid = gce.MultiCircuit()

bus1 = gce.Bus(name="Bus1", Vnom=10)
bus2 = gce.Bus(name="Bus2", Vnom=10)
grid.add_bus(bus1)
grid.add_bus(bus2)

line = gce.Line(name="line 1-2", bus_from=bus1, bus_to=bus2,
                r=0.029585798816568046, x=0.07100591715976332, b=0.03, rate=100.0)
grid.add_line(line)

gen = gce.Generator(name="Gen1", P=10, vset=1.0) # PV
grid.add_generator(bus=bus1, api_obj=gen)

load = gce.Load(name="Load1", P=10, Q=10)        # PQ
grid.add_load(bus=bus2, api_obj=load)

res = gce.power_flow(grid)

print(f"Converged: {res.converged}")

# System
v1 = res.voltage[0]
v2 = res.voltage[1]

Sb1 = res.Sbus[0] / grid.Sbase
Sb2 = res.Sbus[1] / grid.Sbase
Sf = res.Sf / grid.Sbase
St = res.St / grid.Sbase

# Generator
# Current from power and voltage
i = np.conj(Sb1 / v1)          # ī = (p - jq) / v̄*
# Delta angle 
delta0 = np.angle(v1 + ra.value + 1j*xd.value * i)
# dq0 rotation
rot = np.exp(-1j * (delta0 - np.pi/2))
# dq voltages and currents
v_d0 = np.real(v1*rot)
v_q0 = np.imag(v1*rot)
i_d0 = np.real(i*rot)
i_q0 = np.imag(i*rot)
# inductances 
psid0 = -ra.value * i_q0 + v_q0
psiq0 = -ra.value * i_d0 + v_d0

vf0 = - i_d0 + psid0 + xd.value * i_d0
print(f"vf = {vf0}")

mapping = {
    "dline_from": np.angle(v1),
    "dline_to":   np.angle(v2),
    "Vline_from": np.abs(v1),
    "Vline_to":   np.abs(v2),
    "Vg":  np.abs(v1),
    "dg":  np.angle(v1),
    "Pline_from": Sf.real,      # notice .real here
    "Qline_from": Sf.imag,
    "Pline_to":   St.real,
    "Qline_to":   St.imag,
    "Pl": Sb2.real,
    "Ql": Sb2.imag,
    "delta": delta0,
    "omega": 1.0,
    "psid": psid0,
    "psiq": psiq0,
    "i_d": i_d0,
    "i_q": i_q0,
    "v_d": v_d0,
    "v_q": v_q0,
    "t_e": 0.1,
    "p_g": Sb1.real,
    "Q_g": Sb1.imag,
}

# ----------------------------------------------------------------------------------------------------------------------
# Check
# ----------------------------------------------------------------------------------------------------------------------
def line_power_terms(m, g, b, bsh):
    d_f, d_t = m["dline_from"], m["dline_to"]
    Vf,  Vt  = m["Vline_from"], m["Vline_to"]
    
    P_from = (Vf**2 * g.value
              - g.value * Vf * Vt * np.cos(d_f - d_t)
              + b.value * Vf * Vt * np.cos(d_f - d_t + np.pi/2))
    
    Q_from = (Vf**2 * (-bsh.value/2 - b.value)
              - g.value * Vf * Vt * np.sin(d_f - d_t)
              + b.value * Vf * Vt * np.sin(d_f - d_t + np.pi/2))
    
    P_to   = (Vt**2 * g.value
              - g.value * Vt * Vf * np.cos(d_t - d_f)
              + b.value * Vt * Vf * np.cos(d_t - d_f + np.pi/2))
    
    Q_to   = (Vt**2 * (-bsh.value/2 - b.value)
              - g.value * Vt * Vf * np.sin(d_t - d_f)
              + b.value * Vt * Vf * np.sin(d_t - d_f + np.pi/2))
    return P_from, Q_from, P_to, Q_to

P_from, Q_from, P_to, Q_to = line_power_terms(mapping, g, b, bsh)

print("P_from =", P_from)
print("Q_from =", Q_from)
print("P_to   =", P_to)
print("Q_to   =", Q_to)

# ----------------------------------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------------------------------
slv = BlockSolver(sys)

#TODO: run initialization

x0 = slv.build_init_vector(mapping)

events = slv.build_init_events_vector(mapping)
vars_in_order = slv.sort_vars(mapping)

t, y = slv.simulate(
    t0=0,
    t_end=0.1,
    h=0.001,
    x0=x0,
    events=events,
    method="implicit_euler"
)

fig = plt.figure(figsize=(12, 8))
# plt.plot(t, y)
plt.plot(t, y[:, slv.get_var_idx(omega)], label="ω (pu)")
plt.plot(t, y[:, slv.get_var_idx(delta)], label="δ (rad)")
# plt.plot(t, y[:, slv.get_var_idx(t_e)], label="t_e (pu)")
plt.plot(t, y[:, slv.get_var_idx(Vline_from)], label="Vline_from (Vlf)")
plt.plot(t, y[:, slv.get_var_idx(Vline_to)], label="Vline_to (Vlt)")
plt.legend()
plt.show()
