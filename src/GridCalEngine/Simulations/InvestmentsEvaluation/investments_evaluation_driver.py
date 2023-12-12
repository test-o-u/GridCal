# GridCal
# Copyright (C) 2015 - 2023 Santiago Peñate Vera
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
import numpy as np
import hyperopt
import functools

from typing import List, Dict, Union
from GridCalEngine.Simulations.driver_template import DriverTemplate
from GridCalEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver, PowerFlowOptions
from GridCalEngine.Simulations.driver_types import SimulationTypes
from GridCalEngine.Simulations.InvestmentsEvaluation.investments_evaluation_results import InvestmentsEvaluationResults
from GridCalEngine.Core.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Core.Devices.Aggregation.investment import Investment
from GridCalEngine.Core.DataStructures.numerical_circuit import NumericalCircuit
from GridCalEngine.Core.DataStructures.numerical_circuit import compile_numerical_circuit_at
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import multi_island_pf_nc
from GridCalEngine.Simulations.InvestmentsEvaluation.MVRSM import MVRSM_minimize
from GridCalEngine.Simulations.InvestmentsEvaluation.stop_crits import StochStopCriterion
from GridCalEngine.basic_structures import IntVec
from GridCalEngine.enumerations import InvestmentEvaluationMethod
from GridCalEngine.Simulations.InvestmentsEvaluation.investments_evaluation_options import InvestmentsEvaluationOptions
import GridCalEngine.Core.Compilers.circuit_to_newton_pa as newton_interface


class InvestmentsEvaluationDriver(DriverTemplate):
    name = 'Investments evaluation'
    tpe = SimulationTypes.InvestmestsEvaluation_run

    def __init__(self, grid: MultiCircuit,
                 options: InvestmentsEvaluationOptions):
        """
        InputsAnalysisDriver class constructor
        :param grid: MultiCircuit instance
        :param method: InvestmentEvaluationMethod
        :param max_eval: Maximum number of evaluations
        """
        DriverTemplate.__init__(self, grid=grid)
        self.newton_grid, _ = newton_interface.to_newton_pa(grid,
                                                            use_time_series=False,
                                                            time_indices=None,
                                                            override_branch_controls=options.pf_options.override_branch_controls,
                                                            opf_results=None)
        self.newton_pf_options = newton_interface.get_newton_pa_pf_options(options.pf_options)

        # options object
        self.options = options

        # results object
        self.results = InvestmentsEvaluationResults(investment_groups_names=grid.get_investment_groups_names(),
                                                    max_eval=0)

        self.__eval_index = 0

        # dictionary of investment groups
        self.investments_by_group: Dict[int, List[Investment]] = self.grid.get_investmenst_by_groups_index_dict()

        # dimensions
        self.dim = len(self.grid.investments_groups)

        self.capex_factor = 1
        self.l_factor = 1
        self.oload_factor = 1
        self.vm_factor = 1

    def get_steps(self):
        """

        :return:
        """
        return self.results.get_index()

    def objective_function(self, combination: IntVec):
        """
        Function to evaluate a combination of investments
        :param combination: vector of investments (yes/no). Length = number of investment groups
        :return: objective function value
        """

        # add all the investments of the investment groups reflected in the combination
        inv_list = list()
        for i, active in enumerate(combination):
            if active == 1:
                inv_list += self.investments_by_group[i]

        grid_copy = self.newton_grid.copy()

        # gather a dictionary of all the elements, this serves for the investments generation

        all_elements_dict = newton_interface.get_all_elements_dict(grid_copy)

        # enable the investment
        newton_interface.set_investments_status(investments_list=inv_list,
                                                status=True,
                                                all_elemnts_dict=all_elements_dict)

        res = newton_interface.npa.runPowerFlow(circuit=self.newton_grid,
                                                pf_options=self.newton_pf_options,
                                                time_indices=[0],
                                                n_threads=1,
                                                V0=None)

        branches = self.grid.get_branches_wo_hvdc()
        buses = self.grid.get_buses()

        norm = False

        overload_score, oload_limits = get_overload_score(res.Loading[0, :], branches, norm)
        voltage_module_score, vm_limits = get_voltage_module_score(res.voltage[0, :], buses, norm)
        voltage_angle_score = 0.0
        voltage_angle_score, va_limits = get_voltage_phase_score(res.voltage[0, :], buses, norm)
        capex_array = np.array([inv.CAPEX for inv in inv_list])
        if self.__eval_index == 0:
            capex_array = np.array([0])

        losses_score = np.sum(res.Losses.real[0, :])
        losses_limits = (np.max(res.Losses.real[0, :]), np.min(res.Losses.real[0, :]))
        capex_score = np.sum(capex_array)
        capex_limits = (np.max(capex_array), np.min(capex_array))

        # opex_score = get_opex_score()

        if self.__eval_index == 1:
            self.l_factor, self.oload_factor, self.vm_factor, self.capex_factor = get_scale_factors(
                [losses_limits, oload_limits, vm_limits, capex_limits])

        f = (losses_score * self.l_factor * self.options.w_losses +
             overload_score * self.oload_factor * self.options.w_overload +
             voltage_module_score * self.vm_factor * self.options.w_voltage_module +
             capex_score * self.capex_factor * self.options.w_capex)

        # store the results
        self.results.set_at(eval_idx=self.__eval_index,
                            capex=capex_score * self.capex_factor,
                            opex=sum([inv.OPEX for inv in inv_list]),
                            losses=losses_score * self.l_factor,
                            overload_score=overload_score,
                            voltage_score=voltage_module_score,
                            objective_function=f,
                            combination=combination,
                            index_name="Evaluation {}".format(self.__eval_index))

        # increase evaluations
        self.__eval_index += 1

        self.progress_signal.emit(self.__eval_index / self.options.max_eval * 100.0)

        # print(losses_score*self.l_factor, capex_score * self.capex_factor, f)
        return f

    def independent_evaluation(self) -> None:
        """
        Run a one-by-one investment evaluation without considering multiple evaluation groups at a time
        """
        self.results = InvestmentsEvaluationResults(investment_groups_names=self.grid.get_investment_groups_names(),
                                                    max_eval=len(self.grid.investments_groups) + 1)

        # evaluate the investments
        self.__eval_index = 0

        # add baseline
        self.objective_function(combination=np.zeros(self.results.n_groups, dtype=int))

        dim = len(self.grid.investments_groups)

        for k in range(dim):
            self.progress_text.emit("Evaluating investment group {}...".format(k))

            combination = np.zeros(dim, dtype=int)
            combination[k] = 1

            self.objective_function(combination=combination)

        self.progress_text.emit("Done!")
        self.progress_signal.emit(0.0)

    def optimized_evaluation_hyperopt(self) -> None:
        """
        Run an optimized investment evaluation without considering multiple evaluation groups at a time
        """

        # configure hyperopt:

        # number of random evaluations at the beginning
        rand_evals = round(self.dim * 1.5)

        # binary search space
        space = [hyperopt.hp.randint(f'x_{i}', 2) for i in range(self.dim)]

        if self.options.max_eval == rand_evals:
            algo = hyperopt.rand.suggest
        else:
            algo = functools.partial(hyperopt.tpe.suggest, n_startup_jobs=rand_evals)

        self.results = InvestmentsEvaluationResults(investment_groups_names=self.grid.get_investment_groups_names(),
                                                    max_eval=self.options.max_eval + 1)

        # evaluate the investments
        self.__eval_index = 0

        # add baseline
        self.objective_function(combination=np.zeros(self.results.n_groups, dtype=int))

        hyperopt.fmin(self.objective_function, space, algo, self.options.max_eval)

        self.progress_text.emit("Done!")
        self.progress_signal.emit(0.0)

    def optimized_evaluation_mvrsm(self) -> None:
        """
        Run an optimized investment evaluation without considering multiple evaluation groups at a time
        """

        # configure MVRSM:

        # number of random evaluations at the beginning
        rand_evals = round(self.dim * 1.5)
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)
        rand_search_active_prob = 0.5
        threshold = 0.001
        conf_dist = 0.0
        conf_level = 0.95
        stop_crit = StochStopCriterion(conf_dist, conf_level)
        x0 = np.random.binomial(1, rand_search_active_prob, self.dim)

        self.results = InvestmentsEvaluationResults(investment_groups_names=self.grid.get_investment_groups_names(),
                                                    max_eval=self.options.max_eval + 1)

        # evaluate the investments
        self.__eval_index = 0

        # add baseline
        self.objective_function(combination=np.zeros(self.results.n_groups, dtype=int))

        # optimize
        best_x, inv_scale, model = MVRSM_minimize(obj_func=self.objective_function,
                                                  x0=x0,
                                                  lb=lb,
                                                  ub=ub,
                                                  num_int=self.dim,
                                                  max_evals=self.options.max_eval,
                                                  rand_evals=rand_evals,
                                                  obj_threshold=threshold,
                                                  stop_crit=stop_crit,
                                                  rand_search_bias=rand_search_active_prob)

        self.progress_text.emit("Done!")
        self.progress_signal.emit(0.0)

    def run(self):
        """
        run the QThread
        :return:
        """

        self.tic()

        if self.options.solver == InvestmentEvaluationMethod.Independent:
            self.independent_evaluation()

        elif self.options.solver == InvestmentEvaluationMethod.Hyperopt:
            self.optimized_evaluation_hyperopt()

        elif self.options.solver == InvestmentEvaluationMethod.MVRSM:
            self.optimized_evaluation_mvrsm()

        else:
            raise Exception('Unsupported method')

        self.toc()

    def cancel(self):
        self.__cancel__ = True


def get_overload_score(loading, branches, norm):
    branches_cost = np.array([e.Cost for e in branches], dtype=float)
    branches_loading = np.abs(loading)

    # get lines where loading is above 1 -- why not 0.9 ?
    branches_idx = np.where(branches_loading > 1)[0]

    # multiply by the load or only the overload?
    cost = branches_cost[branches_idx] * branches_loading[branches_idx]

    max_oload = 0
    min_oload = 0

    if norm:
        return get_normalized_score(cost), (max_oload, min_oload)
    return np.sum(cost), (max_oload, min_oload)


def get_voltage_module_score(voltage, buses, norm):
    bus_cost = np.array([e.voltage_module_cost for e in buses], dtype=float)
    vmax = np.array([e.Vmax for e in buses], dtype=float)
    vmin = np.array([e.Vmin for e in buses], dtype=float)
    vm = np.abs(voltage)
    vmax_diffs = np.array(vm - vmax).clip(min=0)
    vmin_diffs = np.array(vmin - vm).clip(min=0)
    cost = (vmax_diffs + vmin_diffs) * bus_cost

    if norm:
        return get_normalized_score(cost), (np.max(cost), np.min(cost))
    return np.sum(cost), (np.max(cost), np.min(cost))


def get_voltage_phase_score(voltage, buses, norm):
    bus_cost = np.array([e.voltage_angle_cost for e in buses], dtype=float)
    vpmax = np.array([e.angle_max for e in buses], dtype=float)
    vpmin = np.array([e.angle_min for e in buses], dtype=float)
    vp = np.abs(voltage)
    vpmax_diffs = np.array(vp - vpmax).clip(min=0)
    vpmin_diffs = np.array(vpmin - vp).clip(min=0)
    cost = (vpmax_diffs + vpmin_diffs) * bus_cost

    if norm:
        return get_normalized_score(cost), (np.max(cost), np.min(cost))
    return np.sum(cost), (np.max(cost), np.min(cost))


def get_opex_score(inv_list):
    for inv in inv_list:
        opex = inv.OPEX


def get_normalized_score(array):
    if len(array) < 1:
        return 0.0

    max_value = np.max(array)

    if max_value != 0:
        return np.sum(array) / max_value

    return 0.0


def get_scale_factors(list):
    med = np.zeros(len(list))
    max_values, min_values = zip(*list)
    for i, tpe in enumerate(list):
        med[i] = (tpe[0] + tpe[1]) / 2

    min_med = np.min(med[med != 0])

    med[med == 0] = min_med

    return min_med / med[0], min_med / med[1], min_med / med[2], min_med / med[3]
