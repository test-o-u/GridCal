import numpy

from numpy import *

def f_update(D, M, fn, omega, te, tm):
    return array([[2*pi*fn*(omega - 1)], [-D*(omega - 1)/M + te/M - tm/M]])


def g_update(Pe, Qe, a, delta, i_d, i_q, psid, psiq, ra, te, v, vd, vf, vq, xd):
    return array([[-i_q*ra - psid + vq], [-i_d*ra - psiq + vd], [i_d*xd + psid - vf], [i_q*xd + psiq], [-v*sin(a - delta) - vd], [v*cos(a - delta) - vq], [-i_d*psiq + i_q*psid - te], [-Pe + i_d*vd + i_q*vq], [-Qe + i_d*vq - i_q*vd], [i_d*vd + i_q*vq], [i_d*vq - i_q*vd]])


f_args =['D', 'M', 'fn', 'omega', 'te', 'tm']
g_args =['Pe',
 'Qe',
 'a',
 'delta',
 'i_d',
 'i_q',
 'psid',
 'psiq',
 'ra',
 'te',
 'v',
 'vd',
 'vf',
 'vq',
 'xd']
variables_names_for_ordering ={'f': ['delta', 'omega'],
 'g': ['psid', 'psiq', 'i_d', 'i_q', 'vd', 'vq', 'te', 'Pe', 'Qe', 'a', 'v']}
def f_ia(D, M, fn):
    return (0, 2*pi*fn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -D/M, 0, 0, 0, 0, 0, 0, M**(-1.0), 0, 0, 0, 0,)


def g_ia(a, delta, i_d, i_q, psid, psiq, ra, v, vd, vq, xd):
    return (0, 0, -1, 0, 0, -ra, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -ra, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, xd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, xd, 0, 0, 0, 0, 0, 0, 0, v*cos(a - delta), 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -v*cos(a - delta), -sin(a - delta), v*sin(a - delta), 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -v*sin(a - delta), cos(a - delta), 0, 0, i_q, -i_d, -psiq, psid, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, vd, vq, i_d, i_q, 0, -1, 0, 0, 0, 0, 0, 0, 0, vq, -vd, -i_q, i_d, 0, 0, -1, 0, 0, 0, 0, 0, 0, vd, vq, i_d, i_q, 0, 0, 0, 0, 0, 0, 0, 0, 0, vq, -vd, -i_q, i_d, 0, 0, 0, 0, 0,)


f_jac_args =['D', 'M', 'fn']
g_jac_args =['a', 'delta', 'i_d', 'i_q', 'psid', 'psiq', 'ra', 'v', 'vd', 'vq', 'xd']
jacobian_info = {'dfx': [(0, 1), (1, 1)], 'dfy': [(1, 8)], 'dgx': [(6, 0), (7, 0)], 'dgy': [(2, 2), (2, 5), (2, 7), (3, 3), (3, 4), (3, 6), (4, 2), (4, 4), (5, 3), (5, 5), (6, 6), (6, 11), (6, 12), (7, 7), (7, 11), (7, 12), (8, 2), (8, 3), (8, 4), (8, 5), (8, 8), (9, 4), (9, 5), (9, 6), (9, 7), (9, 9), (10, 4), (10, 5), (10, 6), (10, 7), (10, 10), (11, 4), (11, 5), (11, 6), (11, 7), (12, 4), (12, 5), (12, 6), (12, 7)]}