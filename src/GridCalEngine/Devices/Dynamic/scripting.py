# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb

from GridCalEngine import *
import matplotlib.pyplot as plt
from typing import List

from GridCalEngine.Devices.Dynamic.main import start_dynamic
from GridCalEngine.Devices.Dynamic.models.dynmodel import DynamicModel
from GridCalEngine.Devices.Parents.branch_parent import BranchParent
from GridCalEngine.Devices.Parents.injection_parent import InjectionParent
from GridCalEngine.Simulations.Dynamic.dinamic_driver import DynamicDriver

# grid data, this data will be automatically generated when the user builds the grid.

bus1_data = {
      "name": "Bus",
      "comp_code": [0],
      "comp_name": ["Bus1"],
      "u": [1],
      "idx_dyn_param": [],
      "num_dyn_param": [{"name":"p0",
                             "symbol":"p0",
                              "info":"initial voltage phase angle",
                              "value":[0]},
                        {"name":"q0",
                             "symbol":"q0",
                              "info":"initial voltage magnitude",
                              "value":[1]}],
      "ext_dyn_param": [],
      "stat_var": [],
      "algeb_var": [{"name":"p",
                             "symbol":"p",
                             "init_eq":"",
                             "eq":""},
                    {"name":"q",
                             "symbol":"q",
                             "init_eq":"",
                             "eq":""}],
      "ext_state_var": [],
      "ext_algeb_var": []}

bus2_data = {
      "name": "Bus",
      "comp_code": [1],
      "comp_name": ["Bus2"],
      "u": [1],
      "idx_dyn_param": [],
      "num_dyn_param": [{"name":"p0",
                             "symbol":"p0",
                              "info":"initial voltage phase angle",
                              "value":[0]},
                        {"name":"q0",
                             "symbol":"q0",
                              "info":"initial voltage magnitude",
                              "value":[1]}],
      "ext_dyn_param": [],
      "stat_var": [],
      "algeb_var": [{"name":"p",
                             "symbol":"p",
                             "init_eq":"",
                             "eq":""},
                    {"name":"q",
                             "symbol":"q",
                             "init_eq":"",
                             "eq":""}],
      "ext_state_var": [],
      "ext_algeb_var": []}


branch_data = {
      "name": "ACLine",
      "comp_code": [0],
      "comp_name": ["Line1"],
      "u": [1],
      "idx_dyn_param": [{"name": "bus1",
                         "symbol":"Bus",
                         "info":"idx of from bus",
                         "ident":[0],
                         "connection_point":"ACLine_origin"},
                        {"name": "bus2",
                         "symbol":"Bus",
                         "info":"idx of to bus",
                         "ident":[1],
                         "connection_point":"ACLine_end"}],
      "num_dyn_param": [{"name": "g",
                              "symbol":"g",
                              "info":"shared shunt conductance",
                              "value":[0.09]},
                        {"name": "b",
                              "symbol":"b",
                              "info":"shared shunt susceptance",
                              "value":[-20.99]},
                        {"name": "bsh",
                              "symbol":"bsh",
                              "info":"from/to-side shunt susceptance",
                              "value":[-0.000001]}],
      "ext_dyn_param": [],
      "stat_var": [],
      "algeb_var": [],
      "ext_state_var": [],
      "ext_algeb_var": [{"name":"P_origin",
                             "symbol":"P_origin",
                             "src":"p",
                             "indexer": "bus1",
                             "init_eq":"",
                             "eq":"(Q_origin ** 2 * g  - Q_origin * Q_end * (g * cos(P_origin - P_end) + b * sin(P_origin - P_end)))"},
                    {"name":"Q_origin",
                             "symbol":"Q_origin",
                             "src":"q",
                             "indexer": "bus1",
                             "init_eq":"",
                             "eq":"(- Q_origin ** 2 * (b + bsh / 2) - Q_origin * Q_end * (g * sin(P_origin - P_end) - b * cos(P_origin - P_end)))"},
                    {"name":"P_end",
                             "symbol":"P_end",
                             "src":"p",
                             "indexer": "bus2",
                             "init_eq":"",
                             "eq":"(Q_end ** 2 * g  - Q_end * Q_origin * (g * cos(P_end - P_origin) + b * sin(P_end - P_origin)))"},
                    {"name":"Q_end",
                             "symbol":"Q_end",
                             "src":"q",
                             "indexer": "bus2",
                             "init_eq":"",
                             "eq":"(- Q_end ** 2 * (b + bsh / 2) - Q_end * Q_origin * (g * sin(P_end - P_origin) - b * cos(P_end - P_origin)))"}]}

slack_gen_data = {
      "name": "Slack",
      "comp_code": [0],
      "comp_name": ["Slack1"],
      "u": [1],
      "idx_dyn_param": [{"name": "bus",
                               "symbol":"Bus",
                               "info":"interface bus id",
                               "ident":[0],
                               "connection_point":"Slack"}],
      "num_dyn_param": [{"name":"Sn",
                             "symbol":"Sn",
                              "info":"",
                              "value":[1.0]},
                        {"name":"Vn",
                             "symbol":"Vn",
                              "info":"",
                              "value":[1.0]},
                        {"name":"P_e0",
                             "symbol":"P_e0",
                              "info":"",
                              "value":[1.0]},
                        {"name":"Q_e0",
                             "symbol":"Q_e0",
                              "info":"",
                              "value":[1.0]},
                        {"name":"pmax",
                             "symbol":"pmax",
                              "info":"",
                              "value":[0.3]},
                        {"name":"pmin",
                             "symbol":"pmin",
                              "info":"",
                              "value":[0.86138701]},
                        {"name":"qmax",
                             "symbol":"qmax",
                              "info":"",
                              "value":[50]},
                        {"name":"qmin",
                             "symbol":"qmin",
                              "info":"",
                              "value":[50]},
                        {"name":"q0",
                             "symbol":"q0",
                              "info":"",
                              "value":[10]},
                        {"name":"vmax",
                             "symbol":"vmax",
                              "info":"",
                              "value":[1.0]},
                        {"name":"vmin",
                             "symbol":"vmin",
                              "info":"",
                              "value":[0.003]},
                        {"name":"ra",
                             "symbol":"ra",
                              "info":"",
                              "value":[0.3]},
                        {"name":"xs",
                             "symbol":"xs",
                              "info":"",
                              "value":[0.86138701]},
                        {"name":"p0",
                             "symbol":"p0",
                              "info":"",
                              "value":[0.86138701]}],

      "ext_dyn_param": [{"name":"busv0",
                             "symbol":"busv0",
                              "info":"",
                              "value":[0.86138701]},
                   {"name":"busa0",
                             "symbol":"busa0",
                              "info":"",
                              "value":[0.86138701]}],

      "stat_var": [],
      "algeb_var": [{"name":"P_e_slack",
                             "symbol":"P_e_slack",
                             "init_eq":"p0",
                             "eq":"p0-p + pmin-P_e_slack + pmax-P_e_slack"},
                    {"name":"Q_e_slack",
                             "symbol":"Q_e_slack",
                             "init_eq":"q0",
                             "eq":"q0-q + qmin-Q_e_slack + qmax-Q_e_slack"}],
      "ext_state_var": [],
      "ext_algeb_var": [{"name":"p",
                             "symbol":"p",
                             "src":"p",
                             "indexer": "bus",
                             "init_eq":"a0+busa0",
                             "eq":"(-p)"},
                    {"name":"q",
                             "symbol":"q",
                             "src":"q",
                             "indexer": "bus",
                             "init_eq":"v0+busv0",
                             "eq":"(-q)"}]}


syn_gen_data = {
      "name": "GENCLS1",
      "comp_code": [0],
      "comp_name": ["GENCLS 1"],
      "u": [1],
      "idx_dyn_param": [{"name": "bus",
                               "symbol":"Bus",
                               "info":"interface bus id",
                               "ident":[0],
                               "connection_point":"GENCLS"}],
      "num_dyn_param": [{"name": "fn",
                              "symbol":"fn",
                              "info":"rated frequency",
                              "value":[50]},
                        {"name": "D",
                              "symbol":"D",
                              "info":"damping coefficient",
                              "value":[10]},
                        {"name": "M",
                              "symbol":"M",
                              "info":"machine start up time (2H)",
                              "value":[1.0]},
                        {"name": "ra",
                              "symbol":"ra",
                              "info":"armature resistance",
                              "value":[0.003]},
                        {"name": "xd",
                              "symbol":"xd",
                              "info":"d-axis transient reactance",
                              "value":[0.3]},
                        {"name": "tm",
                              "symbol":"tm",
                              "info":"uncontrolled mechanical torque",
                              "value":[0.86138701]},
                        {"name": "vf",
                              "symbol":"vf",
                              "info":"uncontrolled exitation voltage",
                              "value":[3.81099313]}],
      "ext_dyn_param": [],
      "stat_var": [{"name":"delta",
                             "symbol":"delta",
                             "init_eq":"delta0",
                             "eq":"(2 * pi * fn) * (omega - 1)",
                             "var_type": "x"},
                    {"name":"omega",
                             "symbol":"omega",
                             "init_eq":"omega0",
                             "eq":"(-tm / M + t_e / M - D / M * (omega - 1))",
                             "var_type": "x"}],
      "algeb_var": [{"name":"psid",
                             "symbol":"psid",
                             "init_eq":"psid0",
                             "eq":"(-ra * i_q + v_q) - psid",
                             "var_type": "y"},
                    {"name":"psiq",
                             "symbol":"psiq",
                             "init_eq":"psiq0",
                             "eq":"(-ra * i_d + v_d) - psiq",
                             "var_type": "y"},
                    {"name":"i_d",
                             "symbol":"i_d",
                             "init_eq":"i_d0",
                             "eq":"psid + xd * i_d - vf",
                             "var_type": "y"},
                    {"name":"i_q",
                             "symbol":"i_q",
                             "init_eq":"i_q0",
                             "eq":"psiq + xd * i_q",
                             "var_type": "y"},
                    {"name":"v_d",
                             "symbol":"v_d",
                             "init_eq":"v_d0",
                             "eq":"q * sin(delta - p) - v_d",
                             "var_type": "y"},
                    {"name":"v_q",
                             "symbol":"v_q",
                             "init_eq":"v_q0",
                             "eq":"q * cos(delta - p) - v_q",
                             "var_type": "y"},
                    {"name":"t_e",
                             "symbol":"t_e",
                             "init_eq":"t_m",
                             "eq":"(psid * i_q - psiq * i_d) - t_e",
                             "var_type": "y"},
                    {"name":"P_e",
                             "symbol":"P_e",
                             "init_eq":"(v_d0 * i_d0 + v_q0 * i_q0)",
                             "eq":"(v_d * i_d + v_q * i_q) - P_e",
                             "var_type": "y"},
                    {"name":"Q_e",
                             "symbol":"Q_e",
                             "init_eq":"(v_q0 * i_d0 - v_d0 * i_q0)",
                             "eq":"(v_q * i_d - v_d * i_q) - Q_e",
                             "var_type": "y"}],
      "ext_state_var": [],
      "ext_algeb_var": [{"name":"p",
                             "symbol":"p",
                             "src":"p",
                             "indexer": "bus",
                             "init_eq":"",
                             "eq":"(v_d * i_d + v_q * i_q)"},
                    {"name":"q",
                             "symbol":"q",
                             "src":"q",
                             "indexer": "bus",
                             "init_eq":"",
                             "eq":"(v_q * i_d - v_d * i_q)"}]}

load_data = {
      "name": "ExpLoad1",
      "comp_code": [0],
      "comp_name": ["ExpLoad 1"],
      "u": [1],
      "idx_dyn_param": [{"name": "bus",
                               "symbol":"Bus",
                               "info":"interface bus id",
                               "ident":[1],
                               "connection_point":"ExpLoad"}],
      "num_dyn_param": [{"name": "coeff_alfa",
                              "symbol":"coeff_alfa",
                              "info":"Active power load exponential coefficient",
                              "value":[0.0]},
                        {"name": "coeff_beta",
                              "symbol":"coeff_beta",
                              "info":"Active power load exponential coefficient",
                              "value":[0.0]},
                        {"name": "Pl0",
                              "symbol":"Pl0",
                              "info":"Active Power load base",
                              "value":[0.099]},
                        {"name": "Ql0",
                              "symbol":"Ql0",
                              "info":"Reactive Power load base",
                              "value":[0.198]}],
      "ext_dyn_param": [],
      "stat_var": [],
      "algeb_var": [],
      "ext_state_var": [],
      "ext_algeb_var": [{"name":"p",
                             "symbol":"p",
                             "src":"p",
                             "indexer": "bus",
                             "init_eq":"",
                             "eq":"Pl0 * q ** coeff_alfa"},
                    {"name":"q",
                             "symbol":"q",
                             "src":"q",
                             "indexer": "bus",
                             "init_eq":"",
                             "eq":"Ql0 * q ** coeff_beta"}]}

# build the grid create models and add devices to the grid

np.set_printoptions(precision=4)
grid = MultiCircuit()

# MyBus1 = DynamicModel()
# MyBus1.parse(bus1_data)
#
# MyBus2 = DynamicModel()
# MyBus2.parse(bus2_data)
#
# MyBranch1 = DynamicModel()
# MyBranch1.parse(branch_data)
#
# MySlackGenerator1 = DynamicModel()
# MySlackGenerator1.parse(slack_gen_data)
#
# MySynGenerator1 = DynamicModel()
# MySynGenerator1.parse(syn_gen_data)
#
# MyLoad1 = DynamicModel()
# MyLoad1.parse(load_data)




# Add the buses and the generators and loads attached
bus1 = Bus('Bus 1', Vnom=20)
bus1.dynamic_model.parse(bus1_data)
grid.add_bus(bus1)

bus2 = Bus('Bus 2', Vnom=20)
bus2.dynamic_model.parse(bus2_data)
grid.add_bus(bus2)

#add branches (Lines in this case)
line1 = Line(bus_from=bus1, bus_to=bus2, name='line 1-2', r=0.05, x=0.11, b=0.02)
line1.dynamic_model.parse(branch_data)
grid.add_line(line1)

gen1 = StaticGenerator(name="slack_gen_1", P=4.0, Q=2)
gen1.dynamic_model.parse(slack_gen_data)
grid.add_static_generator(bus=bus1, api_obj=gen1)

gen2 = Generator('Sync Generator', vset=1.0)
gen2.dynamic_model.parse(syn_gen_data)
grid.add_generator(bus1, api_obj=gen2)

Load1 = Load('load_1', P=40, Q=20)
Load1.dynamic_model.parse(load_data)
grid.add_load(bus2, api_obj=Load1)

# to access elements of the grid:
    #grid._generators
    #grid._loads...
#
# for item in grid.items():
#     pdb.set_trace()
#     dynamic_model = item.dynamic_model
#     dynamic_model_name = dynamic_model.name

# Once the grid is built we can access dynamic models and create the dynamic system

Dynamic_simulation = DynamicDriver(grid)

Dynamic_simulation.run()


#next step is to modify SET to parse models with the new configuration



options = PowerFlowOptions(SolverType.NR, verbose=False)
power_flow = PowerFlowDriver(grid, options)
power_flow.run()

results = power_flow.results
df_bus, df_branch = results.export_all()

print(df_bus)
print(df_branch)

print('\n\n', grid.name)
print('\t|V|:', abs(power_flow.results.voltage))
print('\t|Sbranch|:', abs(power_flow.results.Vbranch))
print('\t|loading|:', abs(power_flow.results.loading) * 100)
print('\terr:', power_flow.results.error)
print('\tConv:', power_flow.results.converged)

grid.plot_graph()
plt.show()
