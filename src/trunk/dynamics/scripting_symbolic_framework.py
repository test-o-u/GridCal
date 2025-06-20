# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import math
import pdb

import numpy as np

import GridCalEngine as gce
from GridCalEngine import RmsSimulationDriver
from GridCalEngine.Devices.Dynamic.equation_2 import Equation
from GridCalEngine.Utils.Symbolic.symbolic import Const, Var, get_jacobian, cos, sin

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

#Bus1

p1 = Var("p1")
q1 = Var("q1")


#Bus2

p2 = Var("p2")
q2 = Var("q2")


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

# Bus1
eq_p1 = Pg+P_origin
eq_q1 = Qg+Q_origin

#Bus2
eq_p2 = Pl+P_end
eq_q2 = Ql+Q_end


# Line
eq_P_origin = (Q_origin ** 2 * g) - (Q_origin * Q_end * (g * cos(P_origin - P_end) + b * sin(P_origin - P_end)))
eq_Q_origin = -Q_origin ** 2 * (b + bsh / Const(2)) - Q_origin * Q_end * (
        g * sin(P_origin - P_end) - b * cos(P_origin - P_end))
eq_P_end = (Q_end ** 2 * g) - (Q_end * Q_origin * (g * cos(P_end - P_origin) + b * sin(P_end - P_origin)))
eq_Q_end = -Q_end ** 2 * (b + bsh / Const(2)) - Q_end * Q_origin * (g * sin(P_end - P_origin) - b * cos(P_end - P_origin))

# Load

eq_Pl = Pl0 * Ql ** coeff_alfa
eq_Ql = Ql0 * Ql ** coeff_beta

# Generator

# State equations
eq_delta = delta - (2 * pi * fn) * (omega - 1)
eq_omega = omega - (-tm / M + t_e / M - D / M * (omega - 1))

# Algebraic equationseq_
eq_psid = psid - ((-ra * i_q + v_q) - psid)
eq_psiq = psiq - ((-ra * i_d + v_d) - psiq)
eq_i_d = i_d - (psid + xd * i_d - vf)
eq_i_q = i_q - (psiq + xd * i_q)
eq_v_d = v_d - (Qg * sin(delta - Pg))
eq_v_q = v_q - (Qg * cos(delta - Pg))
eq_t_e = t_e - ((psid * i_q - psiq * i_d) - t_e)
eq_P_e = P_e - (v_d * i_d + v_q * i_q - P_e)
eq_Q_e = Q_e - (v_q * i_d - v_d * i_q - Q_e)
eq_Pg = Pg - (v_d * i_d + v_q * i_q)
eq_Qg = Qg - (v_q * i_d + v_d * i_q)


# Data to parse for building dynamic models ----------------------------------------------------------------------------------------------------------------------
bus1_data = {
    "name": "Bus",
    "idtag": "46afa8ba9d9d4f769296a972e23facfb",
    "algebraic_equations": [{
            'output': p1,
            'eq': eq_p1
        },{
            'output': q1,
            'eq': eq_q1
        }],
    "state_equations": [],
    "state_var_output": [],
    "algebraic_var_output":[{"name": Pg},
                            {"name": Qg},
                            {"name": P_origin},
                            {"name": Q_origin}],
    "state_var_input": [],
    "algebraic_var_input": [{"name": p1},
                            {"name": q1}]}

bus2_data = {
    "name": "Bus",
    "idtag": "4ff496c4e4d74ad08a9e9ee961238c80",
    "algebraic_equations": [{
            'output': p2,
            'eq': eq_p2
        },{
            'output': q2,
            'eq': eq_q2
        }],
    "state_equations": [],
    "state_var_output": [],
    "algebraic_var_output": [{"name": Pl},
                            {"name": Ql},
                            {"name": P_end},
                            {"name": Q_end}],
    "state_var_input": [],
    "algebraic_var_input":  [{"name": p2},
                            {"name": q2}],}

branch_data = {
    'name': 'ACLine',
    'idtag': 'd9e581da5fcc4fbdb9e44e57c412bed5',
    'algebraic_var_input': [],
    'state_var_input': [],
    'algebraic_var_output': [
        {'name': P_origin},
        {'name': Q_origin},
        {'name': P_end},
        {'name': Q_end}
    ],
    'state_var_output': [],

    'algebraic_equations': [
        {
            'output': P_origin,
            'eq': eq_P_origin
        },
        {
            'output': Q_origin,
            'eq': eq_Q_origin
        },
        {
            'output': P_end,
            'eq': eq_P_end
        },
        {
            'output': Q_end,
            'eq': eq_Q_end
        }
    ],
    'state_equations': [],
    'state_var_output': []
}

# slack_gen_data = {
#     'name': 'Slack',
#     'idtag': '7911d0a6ff0748449171a596591831f9',
#     'algebraic_var_input': [],
#     'state_var_input': [],
#     'algebraic_var_output': [
#         {'name': 'p'},
#         {'name': 'q'},
#         {'name': 'P_e_slack'},
#         {'name': 'Q_e_slack'}
#     ],
#     'state_var_output': [],
#     'algebraic_equations': [
#         {
#             'output': Var(name='p'),
#             'eq': smb.sympify('(-p)')
#         },
#         {
#             'output': Var(name='q'),
#             'eq': smb.sympify('(-q)')
#         },
#         {
#             'output': Var(name='P_e_slack'),
#             'eq': smb.sympify('p0-p + pmin-P_e_slack + pmax-P_e_slack')
#         },
#         {
#             'output': Var(name='Q_e_slack'),
#             'eq': smb.sympify('q0-q + qmin-Q_e_slack + qmax-Q_e_slack')
#         }
#     ],
#     'state_equations': []
# }

load_data = {
    'name': 'ExpLoad',
    'idtag': '718726198865404a8e6c6d0a262aa597',
    'algebraic_var_input': [],
    'state_var_input': [],
    'algebraic_var_output': [
        {'name': Pl},
        {'name': Ql}
    ],
    'state_var_output': [],
    'algebraic_equations': [
        {
            'output': Pl,
            'eq': eq_Pl
        },
        {
            'output': Ql,
            'eq': eq_Ql
        }
    ],
    'state_equations': []
}

bus1_model = gce.DynamicModel()
bus1_model.parse(bus1_data)

bus2_model = gce.DynamicModel()
bus2_model.parse(bus2_data)

branch_model = gce.DynamicModel()
branch_model.parse(branch_data)

# slack_gen_model = gce.DynamicModel()
# slack_gen_model.parse(slack_gen_data)

load_model = gce.DynamicModel()
load_model.parse(load_data)

np.set_printoptions(precision=4)

grid.add_rms_model(bus1_model)
grid.add_rms_model(bus2_model)
grid.add_rms_model(branch_model)
# grid.add_rms_model(slack_gen_model)
grid.add_rms_model(load_model)

# ----------------------------------------------------------------------------------------------------------------------
gen2_rms_model = gce.DynamicModel()
gen2_rms_model.name = "GENCLS"


# State variables and equations
gen2_rms_model.add_state_var_output(delta)
gen2_rms_model.add_state_equations(Equation(delta, eq_delta))

gen2_rms_model.add_state_var_output(omega)
gen2_rms_model.add_state_equations(Equation(omega, eq_omega))

# Algebraic variables and equations
gen2_rms_model.add_algebraic_var_output(psid)
gen2_rms_model.add_algebraic_equations(Equation(psid, eq_psid))

gen2_rms_model.add_algebraic_var_output(psiq)
gen2_rms_model.add_algebraic_equations(Equation(psiq, eq_psiq))

gen2_rms_model.add_algebraic_var_output(i_d)
gen2_rms_model.add_algebraic_equations(Equation(i_d, eq_i_d))

gen2_rms_model.add_algebraic_var_output(i_q)
gen2_rms_model.add_algebraic_equations(Equation(i_q, eq_i_q))

gen2_rms_model.add_algebraic_var_output(v_d)
gen2_rms_model.add_algebraic_equations(Equation(v_d, eq_v_d))

gen2_rms_model.add_algebraic_var_output(v_q)
gen2_rms_model.add_algebraic_equations(Equation(v_q, eq_v_q))

gen2_rms_model.add_algebraic_var_output(t_e)
gen2_rms_model.add_algebraic_equations(Equation(t_e, eq_t_e))

gen2_rms_model.add_algebraic_var_output(P_e)
gen2_rms_model.add_algebraic_equations(Equation(P_e, eq_P_e))

gen2_rms_model.add_algebraic_var_output(Q_e)
gen2_rms_model.add_algebraic_equations(Equation(Q_e, eq_Q_e))

# Output algebraic variables
gen2_rms_model.add_algebraic_var_output(Pg)
gen2_rms_model.add_algebraic_equations(Equation(Pg, eq_Pg))

gen2_rms_model.add_algebraic_var_output(Qg)
gen2_rms_model.add_algebraic_equations(Equation(Qg, eq_Qg))

grid.add_rms_model(gen2_rms_model)

# Assign the dynamic RMS models
bus1.rms_model.template = bus1_model
bus2.rms_model.template = bus2_model
line1.rms_model.template = branch_model
# gen1.rms_model.template = slack_gen_model
gen2.rms_model.template = gen2_rms_model
Load1.rms_model.template = load_model


rms_options = gce.RmsOptions()
Dynamic_simulation = RmsSimulationDriver(grid=grid, options=rms_options)

Dynamic_simulation.run()




#
# equations_list = [eq_psid, eq_psiq, eq_i_d, eq_i_q, eq_v_d, eq_v_q, eq_t_e, eq_P_e, eq_Q_e, eq_Qg,
#                   eq_Pg]
# var_list = [delta, omega, psid, psiq, i_d, i_q, v_q, v_d, t_e, P_e, Q_e, Pg, Qg]
#
#
# # params_list = [fn, tm, M, D, ra, xd, vf]
# params_list = list()
# j_func, _ = get_jacobian(equations=equations_list,
#                          variables=var_list,
#                          params=params_list)
#
# J = j_func(np.ones(len(var_list) + len(params_list)))
#
# print(J.toarray())
