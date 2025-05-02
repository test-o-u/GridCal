# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import pdb

import pandas as pd
import matplotlib.pyplot as plt

from GridCalEngine.Devices.Dynamic.utils.paths import get_andes_results_path

class Data_processor():
    def __init__(self, system):
        # Pass the system object
        self.system = system
        self.variables_list = list()
        self.dataframe = pd.DataFrame()
        self.create_df()

    def create_df(self):
        for element in self.system.dae.internal_variables_list:
            self.variables_list.insert(element[0], element[1])
        self.variables_list.append('time_stamp')
        self.dataframe = pd.DataFrame(columns=self.variables_list)


    def save_iteration(self, iteration_values):

        values_list = list()
        # add states variables values
        for val in iteration_values[1]:
            values_list.append(val)
        # add algebraic variables values
        for val in iteration_values[2]:
            values_list.append(val)
        #add time stamp
        values_list.append(iteration_values[0])

        self.dataframe.loc[len(self.dataframe)] = values_list

    def save_data(self, results):

        for iteration_values in results:
            self.save_iteration(iteration_values)

    def export_csv(self):

        self.dataframe.to_csv('GridCalEngine/Devices/Dynamic/results/system_name.csv')

    def plot_results(self, variables_to_plot = 'all' , time_period = 'full'):

        if variables_to_plot == 'all':
            vars_to_plot = self.variables_list
        else:
            vars_to_plot = variables_to_plot

        if time_period == 'full':
            time_stamps = self.dataframe['time_stamp'][:]

        figure, axis = plt.subplots(3, 5, figsize=(15, 8))
        figure.tight_layout()

        axis[0,0].scatter(self.dataframe['time_stamp'], self.dataframe['delta_GENCLS_0'], marker='.',color='b', linewidths=0.01)
        axis[0, 0].set_title("delta_GENCLS", size=10)
        #axis[0,0].set_xlim([0, 5])
        #axis[0,0].set_ylim([0, -0.5])

        axis[0, 1].scatter(self.dataframe['time_stamp'], self.dataframe['omega_GENCLS_0'], marker='.',color='b', linewidths=0.01)
        axis[0, 1].set_title("omega_GENCLS_0", size=10)
        axis[0, 1].set_ylim([0.9, 1.1])

        axis[0, 2].scatter(self.dataframe['time_stamp'], self.dataframe['psid_GENCLS_0'], marker='.',color='b', linewidths=0.01)
        axis[0, 2].set_title("psid_GENCLS_0", size=10)
        #axis[1, 0].set_ylim([0, -0.5])

        axis[0, 3].scatter(self.dataframe['time_stamp'], self.dataframe['psiq_GENCLS_0'], marker='.',color='b', linewidths=0.01)
        axis[0, 3].set_title("psiq_GENCLS_0", size=10)
        #axis[1, 1].set_ylim([0, -0.5])

        axis[0, 4].scatter(self.dataframe['time_stamp'], self.dataframe['i_q_GENCLS_0'], marker='.', color='b', linewidths=0.01)
        axis[0, 4].set_title("i_q_GENCLS", size=10)
        # axis[0,0].set_xlim([0, 5])
        # axis[0,0].set_ylim([0, -0.5])

        axis[1, 0].scatter(self.dataframe['time_stamp'], self.dataframe['vd_GENCLS_0'], marker='.', color='b', linewidths=0.01)
        axis[1, 0].set_title("vd_GENCLS_0", size=10)
        # axis[0, 1].set_ylim([0.9, 1.1])

        axis[1, 1].scatter(self.dataframe['time_stamp'], self.dataframe['vq_GENCLS_0'], marker='.', color='b', linewidths=0.01)
        axis[1, 1].set_title("vq_GENCLS_0", size=10)
        #axis[0, 1].set_ylim([0.9, 1.1])

        axis[1, 2].scatter(self.dataframe['time_stamp'], self.dataframe['te_GENCLS_0'], marker='.', color='b', linewidths=0.01)
        axis[1, 2].set_title("te_GENCLS_0", size=10)
        # axis[1, 0].set_ylim([0, -0.5])

        axis[1, 3].scatter(self.dataframe['time_stamp'], self.dataframe['Pe_GENCLS_0'], marker='.', color='b', linewidths=0.01)
        axis[1, 3].set_title("Pe_GENCLS_0", size=10)
        # axis[1, 1].set_ylim([0, -0.5])

        axis[1, 4].scatter(self.dataframe['time_stamp'], self.dataframe['Qe_GENCLS_0'], marker='.', color='b', linewidths=0.01)
        axis[1, 4].set_title("Qe_GENCLS_0", size=10)
        # axis[1, 1].set_ylim([0, -0.5])

        axis[2, 0].scatter(self.dataframe['time_stamp'], self.dataframe['a_Bus_0'], marker='.', color='b', linewidths=0.01)
        axis[2, 0].set_title("a_BUS_0", size=10)
        # axis[0, 1].set_ylim([0.9, 1.1])

        axis[2, 1].scatter(self.dataframe['time_stamp'], self.dataframe['v_Bus_0'], marker='.', color='b', linewidths=0.01)
        axis[2, 1].set_title("v_BUS_0", size=10)
        # axis[1, 0].set_ylim([0, -0.5])

        axis[2, 2].scatter(self.dataframe['time_stamp'], self.dataframe['a_Bus_1'], marker='.', color='b', linewidths=0.01)
        axis[2, 2].set_title("a_BUS_1", size=10)
        # axis[1, 1].set_ylim([0, -0.5])

        axis[2, 3].scatter(self.dataframe['time_stamp'], self.dataframe['v_Bus_1'], marker='.', color='b', linewidths=0.01)
        axis[2, 3].set_title("v_BUS_1", size=10)
        # axis[1, 1].set_ylim([0, -0.5])

        # self.dataframe.plot(x="time_stamp", y= ['delta_GENCLS_0'], kind='line', legend=True)
        # self.dataframe.plot(x="time_stamp", y=['delta_GENCLS_0'], kind='line', legend=True)
        # self.dataframe.plot(x="time_stamp", y=['delta_GENCLS_0'], kind='line', legend=True)
        # self.dataframe.plot(x="time_stamp", y=['delta_GENCLS_0'], kind='line', legend=True)

        # Display plot
        figure.suptitle(' Gridcal_dyn: System1 no steady state computation ', fontsize=20)
        plt.show()

    def compare_with_andes(self):

        andes_results_df = pd.read_csv(get_andes_results_path())

        andes_results_df.plot(x="Time [s]", y= ['omega GENCLS 0'], kind='scatter', legend=True)

        # Display plot
        plt.show()












