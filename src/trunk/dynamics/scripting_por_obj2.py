# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb

import matplotlib.pyplot as plt
import numpy as np
import sympy as smb
from sympy import symbols, pi, sin, cos, Symbol

import GridCalEngine as gce
from GridCalEngine.Devices.Dynamic.variable import Var
from GridCalEngine.Devices.Dynamic.equation import Equation
from GridCalEngine.Simulations.Rms.rms_driver import RmsSimulationDriver

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


# Data to parse for building dynamic models ----------------------------------------------------------------------------------------------------------------------
bus1_data = {
    "name": "Bus",
    "idtag": "46afa8ba9d9d4f769296a972e23facfb",
    "algebraic_equations": [],
    "state_equations": [],
    "state_var_output": [],
    "algebraic_var_output": [],
    "state_var_input": [],
    "algebraic_var_input": [{"name": "p1"},
                            {"name": "q1"}]}

bus2_data = {
    "name": "Bus",
    "idtag": "4ff496c4e4d74ad08a9e9ee961238c80",
    "algebraic_equations": [],
    "state_equations": [],
    "state_var_output": [],
    "algebraic_var_output": [],
    "state_var_input": [],
    "algebraic_var_input": [{"name": "p2"},
                            {"name": "q2"}]}

branch_data = {
    'name': 'ACLine',
    'idtag': 'd9e581da5fcc4fbdb9e44e57c412bed5',
    'algebraic_var_input': [],
    'state_var_input': [],
    'algebraic_var_output': [
        {'name': 'P_origin'},
        {'name': 'Q_origin'},
        {'name': 'P_end'},
        {'name': 'Q_end'}
    ],
    'state_var_output': [],

    'algebraic_equations': [
        {
            'output': Var(name='P_origin'),
            'eq': smb.sympify(
                '(Q_origin ** 2 * g  - Q_origin * Q_end * (g * cos(P_origin - P_end) + b * sin(P_origin - P_end)))')
        },
        {
            'output': Var(name='Q_origin'),
            'eq': smb.sympify(
                '(- Q_origin ** 2 * (b + bsh / 2) - Q_origin * Q_end * (g * sin(P_origin - P_end) - b * cos(P_origin - P_end)))')
        },
        {
            'output': Var(name='P_end'),
            'eq': smb.sympify(
                '(Q_end ** 2 * g  - Q_end * Q_origin * (g * cos(P_end - P_origin) + b * sin(P_end - P_origin)))')
        },
        {
            'output': Var(name='Q_end'),
            'eq': smb.sympify(
                '(- Q_end ** 2 * (b + bsh / 2) - Q_end * Q_origin * (g * sin(P_end - P_origin) - b * cos(P_end - P_origin)))')
        }
    ],
    'state_equations': [],
    'state_var_output': []
}

slack_gen_data = {
    'name': 'Slack',
    'idtag': '7911d0a6ff0748449171a596591831f9',
    'algebraic_var_input': [],
    'state_var_input': [],
    'algebraic_var_output': [
        {'name': 'p'},
        {'name': 'q'},
        {'name': 'P_e_slack'},
        {'name': 'Q_e_slack'}
    ],
    'state_var_output': [],
    'algebraic_equations': [
        {
            'output': Var(name='p'),
            'eq': smb.sympify('(-p)')
        },
        {
            'output': Var(name='q'),
            'eq': smb.sympify('(-q)')
        },
        {
            'output': Var(name='P_e_slack'),
            'eq': smb.sympify('p0-p + pmin-P_e_slack + pmax-P_e_slack')
        },
        {
            'output': Var(name='Q_e_slack'),
            'eq': smb.sympify('q0-q + qmin-Q_e_slack + qmax-Q_e_slack')
        }
    ],
    'state_equations': []
}

load_data = {
    'name': 'ExpLoad',
    'idtag': '718726198865404a8e6c6d0a262aa597',
    'algebraic_var_input': [],
    'state_var_input': [],
    'algebraic_var_output': [
        {'name': 'Pl'},
        {'name': 'Ql'}
    ],
    'state_var_output': [],
    'algebraic_equations': [
        {
            'output': Var(name='Pl'),
            'eq': smb.sympify('Pl0 * Ql ** coeff_alfa')
        },
        {
            'output': Var(name='Ql'),
            'eq': smb.sympify('Ql0 * Ql ** coeff_beta')
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

slack_gen_model = gce.DynamicModel()
slack_gen_model.parse(slack_gen_data)

load_model = gce.DynamicModel()
load_model.parse(load_data)

np.set_printoptions(precision=4)

grid.add_rms_model(bus1_model)
grid.add_rms_model(bus2_model)
grid.add_rms_model(branch_model)
grid.add_rms_model(slack_gen_model)
grid.add_rms_model(load_model)

# ----------------------------------------------------------------------------------------------------------------------
gen2_rms_model = gce.DynamicModel()
gen2_rms_model.name = "GENCLS"
# Define symbols
fn = Symbol("fn")
omega_sym = Symbol("omega")
tm = Symbol("tm")
M = Symbol("M")
t_e = Symbol("t_e")
D = Symbol("D")
ra = Symbol("ra")
i_q = Symbol("i_q")
v_q = Symbol("v_q")
i_d = Symbol("i_d")
v_d = Symbol("v_d")
vf = Symbol("vf")
psid_sym = Symbol("psid")
psiq_sym = Symbol("psiq")
Qg = Symbol("Qg")
Pg = Symbol("Pg")
xd = Symbol("xd")

# State variables and equations
gen2_rms_model.add_state_var_output(Var("delta"))
gen2_rms_model.add_state_equations(Equation(Var("delta"), (2 * pi * fn) * (omega_sym - 1)))

gen2_rms_model.add_state_var_output(Var("omega"))
gen2_rms_model.add_state_equations(Equation(Var("omega"), (-tm / M + t_e / M - D / M * (omega_sym - 1))))

# Algebraic variables and equations
gen2_rms_model.add_algebraic_var_output(Var("psid"))
gen2_rms_model.add_algebraic_equations(Equation(Var("psid"), (-ra * i_q + v_q) - psid_sym))

gen2_rms_model.add_algebraic_var_output(Var("psiq"))
gen2_rms_model.add_algebraic_equations(Equation(Var("psiq"), (-ra * i_d + v_d) - psiq_sym))

gen2_rms_model.add_algebraic_var_output(Var("i_d"))
gen2_rms_model.add_algebraic_equations(Equation(Var("i_d"), psid_sym + xd * i_d - vf))

gen2_rms_model.add_algebraic_var_output(Var("i_q"))
gen2_rms_model.add_algebraic_equations(Equation(Var("i_q"), psiq_sym + xd * i_q))

gen2_rms_model.add_algebraic_var_output(Var("v_d"))
gen2_rms_model.add_algebraic_equations(Equation(Var("v_d"), Qg * sin(Var("delta").symbol - Pg) - v_d))

gen2_rms_model.add_algebraic_var_output(Var("v_q"))
gen2_rms_model.add_algebraic_equations(Equation(Var("v_q"), Qg * cos(Var("delta").symbol - Pg) - v_q))

gen2_rms_model.add_algebraic_var_output(Var("t_e"))
gen2_rms_model.add_algebraic_equations(Equation(Var("t_e"), (psid_sym * i_q - psiq_sym * i_d) - t_e))

gen2_rms_model.add_algebraic_var_output(Var("P_e"))
gen2_rms_model.add_algebraic_equations(Equation(Var("P_e"), (v_d * i_d + v_q * i_q) - Var("P_e").symbol))

gen2_rms_model.add_algebraic_var_output(Var("Q_e"))
gen2_rms_model.add_algebraic_equations(Equation(Var("Q_e"), (v_q * i_d - v_d * i_q) - Var("Q_e").symbol))

# Output algebraic variables
gen2_rms_model.add_algebraic_var_output(Var("Pg"))
gen2_rms_model.add_algebraic_equations(Equation(Var("Pg"), v_d * i_d + v_q * i_q))

gen2_rms_model.add_algebraic_var_output(Var("Qg"))
gen2_rms_model.add_algebraic_equations(Equation(Var("Qg"), v_q * i_d + v_d * i_q))

grid.add_rms_model(gen2_rms_model)

# Assign the dynamic RMS models
bus1.rms_model.template = bus1_model
bus2.rms_model.template = bus2_model
line1.rms_model.template = branch_model
# gen1.rms_model.template = slack_gen_model
gen2.rms_model.template = gen2_rms_model
Load1.rms_model.template = load_model

pdb.set_trace()

rms_options = gce.RmsOptions()
Dynamic_simulation = RmsSimulationDriver(grid=grid, options=rms_options)

Dynamic_simulation.run()

# next step is to modify SET to parse models with the new configuration


# options = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
# power_flow = gce.PowerFlowDriver(grid, options)
# power_flow.run()
#
# results = power_flow.results
# df_bus, df_branch = results.export_all()
#
# print(df_bus)
# print(df_branch)
#
# print('\n\n', grid.name)
# print('\t|V|:', abs(power_flow.results.voltage))
# print('\t|Sbranch|:', abs(power_flow.results.Vbranch))
# print('\t|loading|:', abs(power_flow.results.loading) * 100)
# print('\terr:', power_flow.results.error)
# print('\tConv:', power_flow.results.converged)
#
# grid.plot_graph()
# plt.show()
