from numba import njit
from numpy import *

@njit(cache=True)
def f_update():
    return zeros((0, 0))

@njit(cache=True)
def g_update(Pl0, Ql0, coeff_alfa, coeff_beta, q):
    return array([[Pl0*q**coeff_alfa], [Ql0*q**coeff_beta]])

f_args =[]
g_args =['Pl0', 'Ql0', 'coeff_alfa', 'coeff_beta', 'q']

variables_names_for_ordering ={'f': [], 'g': ['p', 'q']}

@njit(cache=True)
def f_ia():
    return ()

@njit(cache=True)
def g_ia(Pl0, Ql0, coeff_alfa, coeff_beta, q):
    return (0, Pl0*coeff_alfa*q**coeff_alfa/q, 0, Ql0*coeff_beta*q**coeff_beta/q,)

f_jac_args =[]
g_jac_args =['Pl0', 'Ql0', 'coeff_alfa', 'coeff_beta', 'q']

jacobian_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': [(0, 1), (1, 1)]}
jacobian_equations ={'dfx': [], 'dfy': [], 'dgx': [], 'dgy': ['Pl0*coeff_alfa*q**coeff_alfa/q', 'Ql0*coeff_beta*q**coeff_beta/q']}