import numpy

from numpy import *

def f_update():
    return array([])


def g_update(Pl0, Ql0, coeff_alfa, coeff_beta, v):
    return array([[Pl0*v**coeff_alfa], [Ql0*v**coeff_beta]])


f_args =[]
g_args =['Pl0', 'Ql0', 'coeff_alfa', 'coeff_beta', 'v']
variables_names_for_ordering ={'f': [], 'g': ['a', 'v']}
def f_ia():
    return ()


def g_ia(Pl0, Ql0, coeff_alfa, coeff_beta, v):
    return (0, Pl0*coeff_alfa*v**coeff_alfa/v, 0, Ql0*coeff_beta*v**coeff_beta/v,)


f_jac_args =[]
g_jac_args =['Pl0', 'Ql0', 'coeff_alfa', 'coeff_beta', 'v']
jacobian_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': [(0, 1), (1, 1)]}