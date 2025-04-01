import numpy

from numpy import *

def f_update(D, fn, omega, te, tm):
    return array([[2*pi*fn*(omega - 1)], [-D*(omega - 1) - te + tm]])


def g_update(Pe, Qe, a, delta, i_d, i_q, psid, psiq, ra, te, v, vd, vf, vq, xq):
    return array([[i_q*ra - psid + vq], [i_d*ra - psid + vd], [i_d*xq + psid - vf], [i_q*xq + psiq], [-v*sin(a - delta) - vd], [v*cos(a - delta) - vq], [-i_d*psiq + i_q*psid - te], [-Pe + i_d*vd + i_q*vq], [-Qe + i_d*vq - i_q*vd], [i_d*vd + i_q*vq], [i_d*vq - i_q*vd]])


f_args =['delta', 'omega']
g_args =['Pe', 'Qe', 'a', 'i_d', 'i_q', 'psid', 'psiq', 'te', 'v', 'vd', 'vq']
def f_ia(D, fn):
    return (0, 2*pi*fn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -D, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,)


def g_ia(a, delta, i_d, i_q, psid, psiq, ra, v, vd, vq, xq):
    return (0, 0, -1, 0, 0, ra, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, ra, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, xq, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, xq, 0, 0, 0, 0, 0, 0, 0, v*cos(a - delta), 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -v*cos(a - delta), -sin(a - delta), v*sin(a - delta), 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -v*sin(a - delta), cos(a - delta), 0, 0, i_q, -i_d, -psiq, psid, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, vd, vq, i_d, i_q, 0, -1, 0, 0, 0, 0, 0, 0, 0, vq, -vd, -i_q, i_d, 0, 0, -1, 0, 0, 0, 0, 0, 0, vd, vq, i_d, i_q, 0, 0, 0, 0, 0, 0, 0, 0, 0, vq, -vd, -i_q, i_d, 0, 0, 0, 0, 0,)


f_jac_args =['D', 'fn']
g_jac_args =['a', 'delta', 'i_d', 'i_q', 'psid', 'psiq', 'ra', 'v', 'vd', 'vq', 'xq']
jacobian_info = {'dfx': [(0, 1), (1, 1)], 'dfy': [(1, 8)], 'dgx': [(4, 0), (5, 0)], 'dgy': [(0, 2), (0, 5), (0, 7), (1, 2), (1, 4), (1, 6), (2, 2), (2, 4), (3, 3), (3, 5), (4, 6), (4, 11), (4, 12), (5, 7), (5, 11), (5, 12), (6, 2), (6, 3), (6, 4), (6, 5), (6, 8), (7, 4), (7, 5), (7, 6), (7, 7), (7, 9), (8, 4), (8, 5), (8, 6), (8, 7), (8, 10), (9, 4), (9, 5), (9, 6), (9, 7), (10, 4), (10, 5), (10, 6), (10, 7)]}