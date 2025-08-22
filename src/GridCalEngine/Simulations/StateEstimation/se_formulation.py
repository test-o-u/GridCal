# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Tuple, List
import numpy as np
from scipy.sparse import hstack as sphs, vstack as spvs, csc_matrix, diags
from GridCalEngine.Utils.Sparse.csc2 import CSC, spsolve_csc, scipy_to_mat
from GridCalEngine.DataStructures.numerical_circuit import NumericalCircuit
from GridCalEngine.Topology.admittance_matrices import AdmittanceMatrices
from GridCalEngine.Simulations.StateEstimation.state_estimation_results import NumericStateEstimationResults
from GridCalEngine.Simulations.StateEstimation.state_estimation_inputs import StateEstimationInput
from GridCalEngine.Simulations.PowerFlow.Formulations.pf_formulation_template import PfFormulationTemplate
from GridCalEngine.Simulations.StateEstimation.state_estimation_options import StateEstimationOptions
from GridCalEngine.Simulations.Derivatives.matpower_derivatives import (dSbus_dV_matpower, dSbr_dV_matpower,
                                                                        dIbr_dV_matpower)

from GridCalEngine.Simulations.PowerFlow.NumericalMethods.common_functions import (compute_fx_error,
                                                                                   power_flow_post_process_nonlinear)

from GridCalEngine.Simulations.PowerFlow.NumericalMethods.common_functions import polar_to_rect
from GridCalEngine.Topology.simulation_indices import compile_types
from GridCalEngine.basic_structures import Vec, IntVec, CxVec, ObjVec, Logger
from GridCalEngine.Utils.Sparse.csc2 import CSC


def compute_jacobian_and_residual(Ybus: csc_matrix, Yf: csc_matrix, Yt: csc_matrix, V: CxVec,
                                  f: IntVec, t: IntVec, Cf: csc_matrix, Ct: csc_matrix,
                                  inputs: StateEstimationInput, pvpq: IntVec,
                                  load_per_bus: CxVec,
                                  fixed_slack: bool):
    """
    Get the arrays for calculation
    :param Ybus: Admittance matrix
    :param Yf: "from" admittance matrix
    :param Yt: "to" admittance matrix
    :param V: Voltages complex vector
    :param f: array of "from" indices of branches
    :param t: array of "to" indices of branches
    :param Cf: Connectivity matrix "from"
    :param Ct: Connectivity matrix "to"
    :param inputs: instance of StateEstimationInput
    :param pvpq: array of pq|pv bus indices
    :param load_per_bus: Array of load per bus in p.u. (used to compute the Pg and Qg measurements)
    :param fixed_slack: if true, the measurements on the slack bus are omitted
    :return: H (jacobian), h (residual), S (power injections)
    """
    n = Ybus.shape[0]

    # compute currents
    I = Ybus @ V
    If = Yf @ V
    It = Yt @ V

    # compute powers
    S = V * np.conj(I)
    Sf = V[f] * np.conj(If)
    St = V[t] * np.conj(It)

    dS_dVa, dS_dVm = dSbus_dV_matpower(Ybus, V)
    dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm = dSbr_dV_matpower(Yf, Yt, V, f, t, Cf, Ct)
    dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm = dIbr_dV_matpower(Yf, Yt, V)

    # slice derivatives
    dP_dVa = dS_dVa[np.ix_(inputs.p_idx, pvpq)].real
    dQ_dVa = dS_dVa[np.ix_(inputs.q_idx, pvpq)].imag
    dPg_dVa = dS_dVa[np.ix_(inputs.pg_idx, pvpq)].real
    dQg_dVa = dS_dVa[np.ix_(inputs.qg_idx, pvpq)].imag
    dPf_dVa = dSf_dVa[np.ix_(inputs.pf_idx, pvpq)].real
    dPt_dVa = dSt_dVa[np.ix_(inputs.pt_idx, pvpq)].real
    dQf_dVa = dSf_dVa[np.ix_(inputs.qf_idx, pvpq)].imag
    dQt_dVa = dSt_dVa[np.ix_(inputs.qt_idx, pvpq)].imag
    dIf_dVa = np.abs(dIf_dVa[np.ix_(inputs.if_idx, pvpq)])
    dIt_dVa = np.abs(dIt_dVa[np.ix_(inputs.it_idx, pvpq)])
    dVm_dVa = csc_matrix(np.zeros((len(inputs.vm_idx), len(pvpq))))
    dVa_dVa = csc_matrix(np.diag(np.ones(n))[np.ix_(inputs.va_idx, pvpq)])

    if fixed_slack:
        # With the fixed slack, we don't need to compute the derivative values for the slack Vm
        dP_dVm = dS_dVm[np.ix_(inputs.p_idx, pvpq)].real
        dQ_dVm = dS_dVm[np.ix_(inputs.q_idx, pvpq)].imag
        dPg_dVm = dS_dVm[np.ix_(inputs.pg_idx, pvpq)].real
        dQg_dVm = dS_dVm[np.ix_(inputs.qg_idx, pvpq)].imag
        dPf_dVm = dSf_dVm[np.ix_(inputs.pf_idx, pvpq)].real
        dPt_dVm = dSt_dVm[np.ix_(inputs.pt_idx, pvpq)].real
        dQf_dVm = dSf_dVm[np.ix_(inputs.qf_idx, pvpq)].imag
        dQt_dVm = dSt_dVm[np.ix_(inputs.qt_idx, pvpq)].imag
        dIf_dVm = np.abs(dIf_dVm[np.ix_(inputs.if_idx, pvpq)])
        dIt_dVm = np.abs(dIt_dVm[np.ix_(inputs.it_idx, pvpq)])
        dVm_dVm = csc_matrix(np.diag(np.ones(n))[np.ix_(inputs.vm_idx, pvpq)])
        dVa_dVm = csc_matrix(np.zeros((len(inputs.va_idx), len(pvpq))))
    else:
        # With the non fixed slack, we need to compute the derivative values for the slack Vm
        dP_dVm = dS_dVm[inputs.p_idx, :].real
        dQ_dVm = dS_dVm[inputs.q_idx, :].imag
        dPg_dVm = dS_dVm[inputs.pg_idx, :].real
        dQg_dVm = dS_dVm[inputs.qg_idx, :].imag
        dPf_dVm = dSf_dVm[inputs.pf_idx, :].real
        dPt_dVm = dSt_dVm[inputs.pt_idx, :].real
        dQf_dVm = dSf_dVm[inputs.qf_idx, :].imag
        dQt_dVm = dSt_dVm[inputs.qt_idx, :].imag
        dIf_dVm = np.abs(dIf_dVm[inputs.if_idx, :])
        dIt_dVm = np.abs(dIt_dVm[inputs.it_idx, :])
        dVm_dVm = csc_matrix(np.diag(np.ones(n))[inputs.vm_idx, :])
        dVa_dVm = csc_matrix(np.zeros((len(inputs.va_idx), n)))

    # pack the Jacobian
    H = spvs([
        sphs([dP_dVa, dP_dVm]),
        sphs([dQ_dVa, dQ_dVm]),
        sphs([dPg_dVa, dPg_dVm]),
        sphs([dQg_dVa, dQg_dVm]),
        sphs([dPf_dVa, dPf_dVm]),
        sphs([dPt_dVa, dPt_dVm]),
        sphs([dQf_dVa, dQf_dVm]),
        sphs([dQt_dVa, dQt_dVm]),
        sphs([dIf_dVa, dIf_dVm]),
        sphs([dIt_dVa, dIt_dVm]),
        sphs([dVm_dVa, dVm_dVm]),
        sphs([dVa_dVa, dVa_dVm])
    ])

    # pack the mismatch vector (calculated estimates in per-unit)
    h = np.r_[
        S[inputs.p_idx].real,  # P
        S[inputs.q_idx].imag,  # Q
        S[inputs.pg_idx].real - load_per_bus[inputs.pg_idx].real,  # Pg
        S[inputs.qg_idx].imag - load_per_bus[inputs.qg_idx].imag,  # Qg
        Sf[inputs.pf_idx].real,  # Pf
        St[inputs.pt_idx].real,  # Pt
        Sf[inputs.qf_idx].imag,  # Qf
        St[inputs.qt_idx].imag,  # Qt
        np.abs(If[inputs.if_idx]),  # If
        np.abs(It[inputs.it_idx]),  # It
        np.abs(V[inputs.vm_idx]),  # Vm
        np.angle(V[inputs.va_idx]),  # Va
    ]

    return H, h, S  # Return Sbus in pu


def get_measurements_and_deviations(se_input: StateEstimationInput, Sbase: float) -> Tuple[Vec, Vec, ObjVec]:
    """
    get_measurements_and_deviations the measurements into "measurements" and "sigma"
    ordering: Pinj, Pflow, Qinj, Qflow, Iflow, Vm
    :param se_input: StateEstimationInput object
    :param Sbase: base power in MVA (i.e. 100 MVA)
    :return: measurements vector in per-unit, sigma vector in per-unit
    """

    nz = se_input.size()
    measurements = np.zeros(nz, dtype=object)
    magnitudes = np.zeros(nz)
    sigma = np.zeros(nz)

    # go through the measurements in order and form the vectors
    k = 0
    for lst in [se_input.p_inj,
                se_input.q_inj,
                se_input.pg_inj,
                se_input.qg_inj,
                se_input.pf_value,
                se_input.pt_value,
                se_input.qf_value,
                se_input.qt_value,
                se_input.if_value,
                se_input.it_value]:
        for m in lst:
            magnitudes[k] = m.get_value_pu(Sbase)
            sigma[k] = m.get_standard_deviation_pu(Sbase)
            measurements[k] = m
            k += 1

    for lst in [se_input.vm_value,
                se_input.va_value]:
        for m in lst:
            magnitudes[k] = m.value
            sigma[k] = m.sigma
            measurements[k] = m
            k += 1

    return magnitudes, sigma, measurements


class StateEstimationFormulation(PfFormulationTemplate):

    def __init__(self, V0: CxVec,
                 nc: NumericalCircuit,
                 se_input: StateEstimationInput,
                 options: StateEstimationOptions,
                 logger: Logger):
        """

        :param V0:
        :param S0:
        :param I0:
        :param Y0:
        :param se_input:
        :param options:
        """
        PfFormulationTemplate.__init__(self, V0=V0, options=options)

        self.nc = nc
        self.adm: AdmittanceMatrices = nc.get_admittance_matrices()
        self.conn = self.nc.get_connectivity_matrices()
        self.se_input = se_input

        S0 = self.nc.get_power_injections_pu()

        self.vd, self.pq, self.pv, self.pqv, self.p, self.no_slack = compile_types(
            Pbus=S0.real,
            types=self.nc.bus_data.bus_types
        )

        self.load_per_bus = self.nc.load_data.get_injections_per_bus() / self.nc.Sbase

        self.z, self.sigma, self.measurements = get_measurements_and_deviations(se_input=se_input, Sbase=nc.Sbase)

        sigma2 = np.power(self.sigma, 2.0)
        W_vec = 1.0 / sigma2
        self.W = diags(W_vec)

        # results update every time
        self.H = None
        self.h = None

    def x2var(self, x: Vec):
        """
        Convert X to decision variables
        :param x: solution vector
        """
        a = len(self.no_slack)

        if self.options.fixed_slack:
            b = a + len(self.no_slack)
            # update the vectors
            self.Va[self.no_slack] = x[0:a]
            self.Vm[self.no_slack] = x[a:b]
        else:
            b = a + self.nc.nbus
            self.Va[self.no_slack] = x[0:a]
            self._Vm = x[a:b]  # yes, all V

    def var2x(self) -> Vec:
        """
        Convert the internal decision variables into the vector
        :return: Vector
        """

        if self.options.fixed_slack:
            return np.r_[
                self.Va[self.no_slack],
                self.Vm[self.no_slack]
            ]
        else:
            return np.r_[
                self.Va[self.no_slack],
                self.Vm  # yes, all Vm
            ]

    def size(self) -> int:
        """
        Size of the jacobian matrix
        :return:
        """
        if self.options.fixed_slack:
            return 2 * len(self.no_slack)
        else:
            return len(self.no_slack) + self.nc.nbus

    def check_error(self, x: Vec) -> Tuple[float, Vec]:
        """
        Check error of the solution without affecting the problem
        :param x: Solution vector
        :return: error
        """
        # update the vectors
        Va = self.Va.copy()
        Vm = self.Vm.copy()

        a = len(self.no_slack)

        if self.options.fixed_slack:
            b = a + len(self.no_slack)
            # update the vectors
            Va[self.no_slack] = x[0:a]
            Vm[self.no_slack] = x[a:b]
        else:
            b = a + self.nc.nbus
            Va[self.no_slack] = x[0:a]
            Vm = x[a:b]  # yes, all V

        # compute the complex voltage
        V = polar_to_rect(Vm, Va)

        # first computation of the jacobian and free term
        H, h, Scalc = compute_jacobian_and_residual(Ybus=self.adm.Ybus,
                                                    Yf=self.adm.Yf,
                                                    Yt=self.adm.Yt,
                                                    V=V,
                                                    f=self.nc.passive_branch_data.F,
                                                    t=self.nc.passive_branch_data.T,
                                                    Cf=self.conn.Cf,
                                                    Ct=self.conn.Ct,
                                                    inputs=self.se_input,
                                                    pvpq=self.no_slack,
                                                    load_per_bus=self.no_slack,
                                                    fixed_slack=self.options.fixed_slack)

        # measurements error (in per-unit)
        dz = self.z - h

        HtW = H.T @ self.W

        _f = HtW @ dz

        # compute the error
        return compute_fx_error(_f), x

    def update(self, x: Vec, update_controls: bool = False) -> Tuple[float, bool, Vec, Vec]:
        """
        Update step
        :param x: Solution vector
        :param update_controls:
        :return: error, converged?, x
        """
        # set the problem state
        self.x2var(x)

        # compute the complex voltage
        self.V = polar_to_rect(self.Vm, self.Va)

        # numerical magic, that avoids negative Vm
        self._Vm = np.abs(self.V)
        self._Va = np.angle(self.V)

        # first computation of the jacobian and residual
        self.H, self.h, Scalc = compute_jacobian_and_residual(Ybus=self.adm.Ybus,
                                                              Yf=self.adm.Yf,
                                                              Yt=self.adm.Yt,
                                                              V=self.V,
                                                              f=self.nc.passive_branch_data.F,
                                                              t=self.nc.passive_branch_data.T,
                                                              Cf=self.conn.Cf,
                                                              Ct=self.conn.Ct,
                                                              inputs=self.se_input,
                                                              pvpq=self.no_slack,
                                                              load_per_bus=self.no_slack,
                                                              fixed_slack=self.options.fixed_slack)

        # measurements error (in per-unit)
        dz = self.z - self.h

        HtW = self.H.T @ self.W

        # compute the function residual
        # this updates H and h
        self._f = HtW @ dz

        # compute the error
        self._error = compute_fx_error(self._f)

        # converged?
        self._converged = self._error < self.options.tolerance

        return self._error, self._converged, x, self.f

    def fx(self) -> Vec:
        return -self._f

    def Jacobian(self) -> CSC:
        """
        Get the Jacobian of the problem, which for SE is not the same
        as the jacobian of the equations, it is a wighted jacobian
        :return:
        """
        # assumes update has been called
        HtW = self.H.T @ self.W
        J = HtW @ self.H

        return scipy_to_mat(J)

    def get_x_names(self) -> List[str]:
        """
        Names matching x
        :return:
        """
        if self.options.fixed_slack:
            cols = [f'dVa {i}' for i in self.no_slack]
            cols += [f'dVm {i}' for i in self.no_slack]
        else:
            cols = [f'dVa {i}' for i in self.no_slack]
            cols += [f'dVm {i}' for i in range(self.nc.nbus)]
        return cols

    def get_fx_names(self) -> List[str]:
        """
        Names matching fx
        :return:
        """
        rows = [f'dP {i}' for i in self.se_input.p_idx]
        rows += [f'dQ {i}' for i in self.se_input.q_inj]

        rows += [f'dPg {i}' for i in self.se_input.pg_inj]
        rows += [f'dQg {i}' for i in self.se_input.qg_inj]

        rows += [f'dPf {i}' for i in self.se_input.pf_idx]
        rows += [f'dPt {i}' for i in self.se_input.pt_idx]
        rows += [f'dQf {i}' for i in self.se_input.qf_idx]
        rows += [f'dQt {i}' for i in self.se_input.qt_idx]

        rows += [f'dIf {i}' for i in self.se_input.if_idx]
        rows += [f'dIt {i}' for i in self.se_input.it_idx]

        rows += [f'dVm {i}' for i in self.se_input.vm_idx]
        rows += [f'dVa {i}' for i in self.se_input.va_idx]

        return rows

    def get_solution(self, elapsed: float, iterations: int) -> NumericStateEstimationResults:
        """
        Get the problem solution
        :param elapsed: Elapsed seconds
        :param iterations: Iteration number
        :return: NumericPowerFlowResults
        """
        # Compute the Branches power and the slack buses power
        Sf, St, If, It, Vbranch, loading, losses, Sbus = power_flow_post_process_nonlinear(
            Sbus=self.Scalc,
            V=self.V,
            F=self.nc.passive_branch_data.F,
            T=self.nc.passive_branch_data.T,
            pv=self.pv,
            vd=self.vd,
            Ybus=self.adm.Ybus,
            Yf=self.adm.Yf,
            Yt=self.adm.Yt,
            Yshunt_bus=self.adm.Yshunt_bus,
            branch_rates=self.nc.passive_branch_data.rates,
            Sbase=self.nc.Sbase
        )

        return NumericStateEstimationResults(V=self.V,
                                             Scalc=Sbus * self.nc.Sbase,
                                             m=self.nc.active_branch_data.tap_module,
                                             tau=self.nc.active_branch_data.tap_angle,
                                             Sf=Sf,
                                             St=St,
                                             If=If,
                                             It=It,
                                             loading=loading,
                                             losses=losses,
                                             Pf_vsc=np.zeros(self.nc.nvsc, dtype=float),
                                             St_vsc=np.zeros(self.nc.nvsc, dtype=complex),
                                             If_vsc=np.zeros(self.nc.nvsc, dtype=float),
                                             It_vsc=np.zeros(self.nc.nvsc, dtype=complex),
                                             losses_vsc=np.zeros(self.nc.nvsc, dtype=float),
                                             loading_vsc=np.zeros(self.nc.nvsc, dtype=float),
                                             Sf_hvdc=np.zeros(self.nc.nhvdc, dtype=complex),
                                             St_hvdc=np.zeros(self.nc.nhvdc, dtype=complex),
                                             losses_hvdc=np.zeros(self.nc.nhvdc, dtype=complex),
                                             loading_hvdc=np.zeros(self.nc.nhvdc, dtype=complex),
                                             norm_f=self.error,
                                             converged=self.converged,
                                             iterations=iterations,
                                             elapsed=elapsed)
