from numba import njit
from numpy import *

@njit(cache=True)
def f_update(D, M, fn, omega, t_e, tm):
    return array([[2*pi*fn*(omega - 1)], [-D*(omega - 1)/M + t_e/M - tm/M]])

@njit(cache=True)
def g_update(P_e, Q_e, delta, i_d, i_q, p, psid, psiq, q, ra, t_e, v_d, v_q, vf, xd):
    return array([[-i_q*ra - psid + v_q], [-i_d*ra - psiq + v_d], [i_d*xd + psid - vf], [i_q*xd + psiq], [q*sin(delta - p) - v_d], [q*cos(delta - p) - v_q], [-i_d*psiq + i_q*psid - t_e], [-P_e + i_d*v_d + i_q*v_q], [-Q_e + i_d*v_q - i_q*v_d], [i_d*v_d + i_q*v_q], [i_d*v_q - i_q*v_d]])

f_args =['D', 'M', 'fn', 'omega', 't_e', 'tm']
g_args =['P_e', 'Q_e', 'delta', 'i_d', 'i_q', 'p', 'psid', 'psiq', 'q', 'ra', 't_e', 'v_d', 'v_q', 'vf', 'xd']

variables_names_for_ordering ={'f': ['delta', 'omega'], 'g': ['psid', 'psiq', 'i_d', 'i_q', 'v_d', 'v_q', 't_e', 'P_e', 'Q_e', 'p', 'q']}

@njit(cache=True)
def f_ia(D, M, fn):
    return (0, 2*pi*fn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -D/M, 0, 0, 0, 0, 0, 0, M**(-1.0), 0, 0, 0, 0,)

@njit(cache=True)
def g_ia(delta, i_d, i_q, p, psid, psiq, q, ra, v_d, v_q, xd):
    return (0, 0, -1, 0, 0, -ra, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -ra, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, xd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, xd, 0, 0, 0, 0, 0, 0, 0, q*cos(delta - p), 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -q*cos(delta - p), sin(delta - p), -q*sin(delta - p), 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, q*sin(delta - p), cos(delta - p), 0, 0, i_q, -i_d, -psiq, psid, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, v_d, v_q, i_d, i_q, 0, -1, 0, 0, 0, 0, 0, 0, 0, v_q, -v_d, -i_q, i_d, 0, 0, -1, 0, 0, 0, 0, 0, 0, v_d, v_q, i_d, i_q, 0, 0, 0, 0, 0, 0, 0, 0, 0, v_q, -v_d, -i_q, i_d, 0, 0, 0, 0, 0,)

f_jac_args =['D', 'M', 'fn']
g_jac_args =['delta', 'i_d', 'i_q', 'p', 'psid', 'psiq', 'q', 'ra', 'v_d', 'v_q', 'xd']

jacobian_info = {'dfx': [(0, 1), (1, 1)], 'dfy': [(1, 8)], 'dgx': [(6, 0), (7, 0)], 'dgy': [(2, 2), (2, 5), (2, 7), (3, 3), (3, 4), (3, 6), (4, 2), (4, 4), (5, 3), (5, 5), (6, 6), (6, 11), (6, 12), (7, 7), (7, 11), (7, 12), (8, 2), (8, 3), (8, 4), (8, 5), (8, 8), (9, 4), (9, 5), (9, 6), (9, 7), (9, 9), (10, 4), (10, 5), (10, 6), (10, 7), (10, 10), (11, 4), (11, 5), (11, 6), (11, 7), (12, 4), (12, 5), (12, 6), (12, 7)]}
jacobian_equations ={'dfx': ['2*pi*fn', '-D/M'], 'dfy': ['1/M'], 'dgx': ['q*cos(delta - p)', '-q*sin(delta - p)'], 'dgy': ['-1', '-ra', '1', '-1', '-ra', '1', '1', 'xd', '1', 'xd', '-1', '-q*cos(delta - p)', 'sin(delta - p)', '-1', 'q*sin(delta - p)', 'cos(delta - p)', 'i_q', '-i_d', '-psiq', 'psid', '-1', 'v_d', 'v_q', 'i_d', 'i_q', '-1', 'v_q', '-v_d', '-i_q', 'i_d', '-1', 'v_d', 'v_q', 'i_d', 'i_q', 'v_q', '-v_d', '-i_q', 'i_d']}