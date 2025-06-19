# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import math

import numpy as np

import GridCalEngine as gce
from GridCalEngine.Utils.Symbolic.symbolic import Const, Var, compile_sparse_jacobian, cos, sin

# grid data, this data will be automatically generated when the user builds the grid.

# instantiate the grid
grid = gce.MultiCircuit(idtag="d3bacc4e2684432991bb3533eff0c453")

# Add the buses, the generators and loads attached and lines
bus1 = gce.Bus('Bus 1', Vnom=20)
grid.add_bus(bus1)

bus2 = gce.Bus('Bus 2', Vnom=20)
grid.add_bus(bus2)

# add branches (Lines in this case)
line1 = gce.Line(bus_from=bus1, bus_to=bus2, name='line 1-2', r=0.05, x=0.11, b=0.02)
grid.add_line(line1)

# add generators
# gen1 = gce.StaticGenerator(name="slack_gen_1", P=4.0, Q=2)
# grid.add_static_generator(bus=bus1, api_obj=gen1)

gen2 = gce.Generator('Sync Generator', vset=1.0)
grid.add_generator(bus1, api_obj=gen2)

# add loads
Load1 = gce.Load('load_1', P=40, Q=20)
grid.add_load(bus2, api_obj=Load1)

# Now the grid is built with the elements attached and we need to add the dynamic model to each element

# First of all the variables are created:
# Define constants

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
fn = Var("fn")
tm = Var("tm")
M = Var("M")
D = Var("D")
ra = Var("ra")
xd = Var("xd")
vf = Var("vf")
# Define variables

# Line
Q_origin = Var("Q_origin")
Q_end = Var("Q_end")
P_origin = Var("P_origin")
P_end = Var("P_end")

# Load
Ql = Var("Ql")
Pl = Var("Pl")

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

var_list = [delta, omega, psid, psiq, i_d, i_q, v_q, v_d, v_q, v_d, t_e, P_e, Q_e, Pg, Qg]
# Build equations

# Line
expr1 = (Q_origin ** 2 * g) - (Q_origin * Q_end * (g * cos(P_origin - P_end) + b * sin(P_origin - P_end)))
expr2 = -Q_origin ** 2 * (b + bsh / Const(2)) - Q_origin * Q_end * (
            g * sin(P_origin - P_end) - b * cos(P_origin - P_end))
expr3 = (Q_end ** 2 * g) - (Q_end * Q_origin * (g * cos(P_end - P_origin) + b * sin(P_end - P_origin)))
expr4 = -Q_end ** 2 * (b + bsh / Const(2)) - Q_end * Q_origin * (g * sin(P_end - P_origin) - b * cos(P_end - P_origin))

# Load

expr5 = Pl0 * Ql ** coeff_alfa
expr6 = Ql0 * Ql ** coeff_beta

# Generator

# State equations
eq_delta = delta - (2 * pi * fn) * (omega - 1)
eq_omega = omega - (-tm / M + t_e / M - D / M * (omega - 1))

# Algebraic equations
expr_psid = psid - ((-ra * i_q + v_q) - psid)
expr_psiq = psiq - ((-ra * i_d + v_d) - psiq)
expr_i_d = i_d - (psid + xd * i_d - vf)
expr_i_q = i_q - (psiq + xd * i_q)
expr_v_d = v_d - (Qg * sin(delta - Pg))
expr_v_q = v_q - (Qg * cos(delta - Pg))
expr_t_e = t_e - ((psid * i_q - psiq * i_d) - t_e)
expr_P_e = P_e - (v_d * i_d + v_q * i_q - P_e)
expr_Q_e = Q_e - (v_q * i_d - v_d * i_q - Q_e)
eq_Pg = Pg - (v_d * i_d + v_q * i_q)
eq_Qg = Qg - (v_q * i_d + v_d * i_q)

equations_list = [expr_psid, expr_psiq, expr_i_d, expr_i_q, expr_v_d, expr_v_q, expr_t_e, expr_P_e, expr_Q_e, eq_Qg,
                  eq_Pg]
var_list = [delta, omega, psid, psiq, i_d, i_q, v_q, v_d, v_q, v_d, t_e, P_e, Q_e, Pg, Qg]

j_func, _ = compile_sparse_jacobian(equations_list, var_list)

J = j_func(np.ones(len(var_list)))

print(J.toarray())
