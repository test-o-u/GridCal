from numba import njit
from numpy import *

@njit(cache=True)
def f_update():
    return zeros((0, 0))

@njit(cache=True)
def g_update(P_e_slack, Q_e_slack, p, p0, pmax, pmin, q, q0, qmax, qmin):
    return array([[-2*P_e_slack - p + p0 + pmax + pmin], [-2*Q_e_slack - q + q0 + qmax + qmin], [-p], [-q]])

f_args =[]
g_args =['P_e_slack', 'Q_e_slack', 'p', 'p0', 'pmax', 'pmin', 'q', 'q0', 'qmax', 'qmin']

variables_names_for_ordering ={'f': [], 'g': ['P_e_slack', 'Q_e_slack', 'p', 'q']}

@njit(cache=True)
def f_ia():
    return ()

@njit(cache=True)
def g_ia():
    return (-2, 0, -1, 0, 0, -2, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1,)

f_jac_args =[]
g_jac_args =[]

jacobian_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': [(0, 0), (0, 2), (1, 1), (1, 3), (2, 2), (3, 3)]}
jacobian_equations ={'dfx': [], 'dfy': [], 'dgx': [], 'dgy': ['-2', '-1', '-2', '-1', '-1', '-1']}