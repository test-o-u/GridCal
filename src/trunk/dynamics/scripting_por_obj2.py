# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import matplotlib.pyplot as plt
import numpy as np

import GridCalEngine as gce
from GridCalEngine.Devices.Dynamic.dyn_var import StatVar, AlgebVar, InputAlgeb
from GridCalEngine.Devices.Dynamic.dyn_param import NumDynParam, IdxDynParam

from GridCalEngine.Simulations.Rms.rms_driver import RmsSimulationDriver

# grid data, this data will be automatically generated when the user builds the grid.

grid = gce.MultiCircuit(idtag="d3bacc4e2684432991bb3533eff0c453")
# Add the buses and the generators and loads attached
bus1 = gce.Bus('Bus 1', Vnom=20)
grid.add_bus(bus1)

bus2 = gce.Bus('Bus 2', Vnom=20)
grid.add_bus(bus2)

# add branches (Lines in this case)
line1 = gce.Line(bus_from=bus1, bus_to=bus2, name='line 1-2', r=0.05, x=0.11, b=0.02)
grid.add_line(line1)

gen1 = gce.StaticGenerator(name="slack_gen_1", P=4.0, Q=2)
grid.add_static_generator(bus=bus1, api_obj=gen1)

gen2 = gce.Generator('Sync Generator', vset=1.0)
grid.add_generator(bus1, api_obj=gen2)

Load1 = gce.Load('load_1', P=40, Q=20)
grid.add_load(bus2, api_obj=Load1)

# ----------------------------------------------------------------------------------------------------------------------

bus1_data = {
    "name": "Bus",
    "idtag": "46afa8ba9d9d4f769296a972e23facfb",
    "comp_code": [0],
    "comp_name": ["Bus 1"],
    "u": [1],
    "idx_dyn_param": [],
    "num_dyn_param": [{"name": "p0",
                       "symbol": "p0",
                       "info": "initial voltage phase angle",
                       "value": [0]},
                      {"name": "q0",
                       "symbol": "q0",
                       "info": "initial voltage magnitude",
                       "value": [1]}],
    "ext_dyn_param": [],
    "stat_var": [],
    "algeb_var": [{"name": "p",
                   "symbol": "p",
                   "init_eq": "",
                   "eq": ""},
                  {"name": "q",
                   "symbol": "q",
                   "init_eq": "",
                   "eq": ""}],
    "input_state_var": [],
    "input_algeb_var": []}

bus1_model = gce.DynamicModel()
bus1_model.parse(bus1_data)

bus2_data = {
    "name": "Bus",
    "idtag": "4ff496c4e4d74ad08a9e9ee961238c80",
    "comp_code": [1],
    "comp_name": ["Bus 2"],
    "u": [1],
    "idx_dyn_param": [],
    "num_dyn_param": [{"name": "p0",
                       "symbol": "p0",
                       "info": "initial voltage phase angle",
                       "value": [0]},
                      {"name": "q0",
                       "symbol": "q0",
                       "info": "initial voltage magnitude",
                       "value": [1]}],
    "ext_dyn_param": [],
    "stat_var": [],
    "algeb_var": [{"name": "p",
                   "symbol": "p",
                   "init_eq": "",
                   "eq": ""},
                  {"name": "q",
                   "symbol": "q",
                   "init_eq": "",
                   "eq": ""}],
    "input_state_var": [],
    "input_algeb_var": []}

bus2_model = gce.DynamicModel()
bus2_model.parse(bus2_data)

branch_data = {
    "name": "ACLine",
    "idtag": 'd9e581da5fcc4fbdb9e44e57c412bed5',
    "comp_code": [0],
    "comp_name": ["Line 1"],
    "u": [1],
    "idx_dyn_param": [],
    "num_dyn_param": [{"name": "g",
                       "symbol": "g",
                       "info": "shared shunt conductance",
                       "value": [0.09]},
                      {"name": "b",
                       "symbol": "b",
                       "info": "shared shunt susceptance",
                       "value": [-20.99]},
                      {"name": "bsh",
                       "symbol": "bsh",
                       "info": "from/to-side shunt susceptance",
                       "value": [-0.000001]}],
    "ext_dyn_param": [],
    "stat_var": [],
    "algeb_var": [],
    "input_state_var": [],
    "input_algeb_var": [{"name": "P_origin",
                       "symbol": "P_origin",
                       "src": "p",
                       "indexer": "bus_from",
                       "init_eq": "",
                       "eq": "(Q_origin ** 2 * g  - Q_origin * Q_end * (g * cos(P_origin - P_end) + b * sin(P_origin - P_end)))"},
                      {"name": "Q_origin",
                       "symbol": "Q_origin",
                       "src": "q",
                       "indexer": "bus_from",
                       "init_eq": "",
                       "eq": "(- Q_origin ** 2 * (b + bsh / 2) - Q_origin * Q_end * (g * sin(P_origin - P_end) - b * cos(P_origin - P_end)))"},
                      {"name": "P_end",
                       "symbol": "P_end",
                       "src": "p",
                       "indexer": "bus_to",
                       "init_eq": "",
                       "eq": "(Q_end ** 2 * g  - Q_end * Q_origin * (g * cos(P_end - P_origin) + b * sin(P_end - P_origin)))"},
                      {"name": "Q_end",
                       "symbol": "Q_end",
                       "src": "q",
                       "indexer": "bus_to",
                       "init_eq": "",
                       "eq": "(- Q_end ** 2 * (b + bsh / 2) - Q_end * Q_origin * (g * sin(P_end - P_origin) - b * cos(P_end - P_origin)))"}]}
branch_model = gce.DynamicModel()
branch_model.parse(branch_data)

slack_gen_data = {
    "name": "Slack",
    "idtag": '7911d0a6ff0748449171a596591831f9',
    "comp_code": [0],
    "comp_name": ["Slack 1"],
    "u": [1],
    "idx_dyn_param": [],
    "num_dyn_param": [{"name": "Sn",
                       "symbol": "Sn",
                       "info": "",
                       "value": [1.0]},
                      {"name": "Vn",
                       "symbol": "Vn",
                       "info": "",
                       "value": [1.0]},
                      {"name": "P_e0",
                       "symbol": "P_e0",
                       "info": "",
                       "value": [1.0]},
                      {"name": "Q_e0",
                       "symbol": "Q_e0",
                       "info": "",
                       "value": [1.0]},
                      {"name": "pmax",
                       "symbol": "pmax",
                       "info": "",
                       "value": [0.3]},
                      {"name": "pmin",
                       "symbol": "pmin",
                       "info": "",
                       "value": [0.86138701]},
                      {"name": "qmax",
                       "symbol": "qmax",
                       "info": "",
                       "value": [50]},
                      {"name": "qmin",
                       "symbol": "qmin",
                       "info": "",
                       "value": [50]},
                      {"name": "q0",
                       "symbol": "q0",
                       "info": "",
                       "value": [10]},
                      {"name": "vmax",
                       "symbol": "vmax",
                       "info": "",
                       "value": [1.0]},
                      {"name": "vmin",
                       "symbol": "vmin",
                       "info": "",
                       "value": [0.003]},
                      {"name": "ra",
                       "symbol": "ra",
                       "info": "",
                       "value": [0.3]},
                      {"name": "xs",
                       "symbol": "xs",
                       "info": "",
                       "value": [0.86138701]},
                      {"name": "p0",
                       "symbol": "p0",
                       "info": "",
                       "value": [0.86138701]}],

    "ext_dyn_param": [{"name": "busv0",
                       "symbol": "busv0",
                       "info": "",
                       "value": [0.86138701]},
                      {"name": "busa0",
                       "symbol": "busa0",
                       "info": "",
                       "value": [0.86138701]}],

    "stat_var": [],
    "algeb_var": [{"name": "P_e_slack",
                   "symbol": "P_e_slack",
                   "init_eq": "p0",
                   "eq": "p0-p + pmin-P_e_slack + pmax-P_e_slack"},
                  {"name": "Q_e_slack",
                   "symbol": "Q_e_slack",
                   "init_eq": "q0",
                   "eq": "q0-q + qmin-Q_e_slack + qmax-Q_e_slack"}],
    "input_state_var": [],
    "input_algeb_var": [{"name": "p",
                       "symbol": "p",
                       "src": "p",
                       "indexer": "bus",
                       "init_eq": "a0+busa0",
                       "eq": "(-p)"},
                      {"name": "q",
                       "symbol": "q",
                       "src": "q",
                       "indexer": "bus",
                       "init_eq": "v0+busv0",
                       "eq": "(-q)"}]}
slack_gen_model = gce.DynamicModel()
slack_gen_model.parse(slack_gen_data)

load_data = {
    "name": "ExpLoad",
    "idtag": '718726198865404a8e6c6d0a262aa597',
    "comp_code": [0],
    "comp_name": ["ExpLoad 1"],
    "u": [1],
    "idx_dyn_param": [],
    "num_dyn_param": [{"name": "coeff_alfa",
                       "symbol": "coeff_alfa",
                       "info": "Active power load exponential coefficient",
                       "value": [0.0]},
                      {"name": "coeff_beta",
                       "symbol": "coeff_beta",
                       "info": "Active power load exponential coefficient",
                       "value": [0.0]},
                      {"name": "Pl0",
                       "symbol": "Pl0",
                       "info": "Active Power load base",
                       "value": [0.099]},
                      {"name": "Ql0",
                       "symbol": "Ql0",
                       "info": "Reactive Power load base",
                       "value": [0.198]}],
    "ext_dyn_param": [],
    "stat_var": [],
    "algeb_var": [],
    "input_state_var": [],
    "input_algeb_var": [{"name": "p",
                       "symbol": "p",
                       "src": "p",
                       "indexer": "bus",
                       "init_eq": "",
                       "eq": "Pl0 * q ** coeff_alfa"},
                      {"name": "q",
                       "symbol": "q",
                       "src": "q",
                       "indexer": "bus",
                       "init_eq": "",
                       "eq": "Ql0 * q ** coeff_beta"}]}
load_model = gce.DynamicModel()
load_model.parse(load_data)

# build the grid create models and add devices to the grid

np.set_printoptions(precision=4)


grid.add_rms_model(bus1_model)
grid.add_rms_model(bus2_model)
grid.add_rms_model(branch_model)
grid.add_rms_model(slack_gen_model)
grid.add_rms_model(load_model)

# ----------------------------------------------------------------------------------------------------------------------
gen2_rms_model = gce.DynamicModel()
gen2_rms_model.name = "GENCLS"
gen2_rms_model.comp_code = [0]
gen2_rms_model.comp_name = ["GENCLS 1"]
gen2_rms_model.u = [1]

gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="fn",
    symbol="fn",
    info="rated frequency",
    value=3.81099313))
gen2_rms_model.add_idx_dyn_param(val=IdxDynParam(
    name="bus",
    symbol="Bus", info="interface bus id",
    ident=[0],
    connection_point="GENCLS"))
gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="D",
    symbol="D",
    info="damping coefficient",
    value=10))
gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="M",
    symbol="M",
    info="machine start up time (2H)",
    value=1.0))
gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="ra",
    symbol="ra",
    info="armature resistance",
    value=0.003))
gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="xd",
    symbol="xd",
    info="d-axis transient reactance",
    value=0.003))
gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="tm",
    symbol="tm",
    info="uncontrolled mechanical torque",
    value=0.3))
gen2_rms_model.add_num_dyn_param(val=NumDynParam(
    name="vf",
    symbol="vf",
    info="uncontrolled excitation voltage",
    value=0.86138701))
gen2_rms_model.add_stat_var(val=StatVar(name="delta",
                                        symbol="delta",
                                        init_eq="delta0",
                                        eq="(2 * pi * fn) * (omega - 1)"))
gen2_rms_model.add_stat_var(val=StatVar(name="omega",
                                        symbol="omega",
                                        init_eq="omega0",
                                        eq="(-tm / M + t_e / M - D / M * (omega - 1))"))
gen2_rms_model.add_algeb_var(val=AlgebVar(name="psid",
                                          symbol="psid",
                                          init_eq="psid0",
                                          eq="(-ra * i_q + v_q) - psid"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="psiq",
                                          symbol="psiq",
                                          init_eq="psiq0",
                                          eq="(-ra * i_d + v_d) - psiq"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="i_d",
                                          symbol="i_d",
                                          init_eq="i_d0",
                                          eq="psid + xd * i_d - vf"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="i_q",
                                          symbol="i_q",
                                          init_eq="i_q0",
                                          eq="psiq + xd * i_q"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="v_d",
                                          symbol="v_d",
                                          init_eq="v_d0",
                                          eq="q * sin(delta - p) - v_d"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="v_q",
                                          symbol="v_q",
                                          init_eq="v_q0",
                                          eq="q * cos(delta - p) - v_q"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="t_e",
                                          symbol="t_e",
                                          init_eq="t_m",
                                          eq="(psid * i_q - psiq * i_d) - t_e"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="P_e",
                                          symbol="P_e",
                                          init_eq="(v_d0 * i_d0 + v_q0 * i_q0)",
                                          eq="(v_d * i_d + v_q * i_q) - P_e"))

gen2_rms_model.add_algeb_var(val=AlgebVar(name="Q_e",
                                          symbol="Q_e",
                                          init_eq="(v_q0 * i_d0 - v_d0 * i_q0)",
                                          eq="(v_q * i_d - v_d * i_q) - Q_e"))
gen2_rms_model.add_input_algeb_var(val=InputAlgeb(name="p",
                                                  symbol="p",
                                                  src="p",
                                                  indexer="bus",
                                                  init_eq="",
                                                  eq="Pl0 * q ** coeff_alfa"))

gen2_rms_model.add_input_algeb_var(val=InputAlgeb(name="q",
                                                  symbol="q",
                                                  src="q",
                                                  indexer="bus",
                                                  init_eq="",
                                                  eq="Ql0 * q ** coeff_beta"))

grid.add_rms_model(gen2_rms_model)


#Assign the dynamic RMS models
bus1.rms_model.template = bus1_model
bus2.rms_model.template = bus2_model
line1.rms_model.template = branch_model
gen1.rms_model.template = slack_gen_model
gen2.rms_model.template = gen2_rms_model
Load1.rms_model.template = load_model
# ----------------------------------------------------------------------------------------------------------------------







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
