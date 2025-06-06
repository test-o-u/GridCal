from numba import njit
from numpy import *

@njit(cache=True)
def f_update():
    return zeros((0, 0))

@njit(cache=True)
def g_update():
    return zeros((0, 0))

f_args =[]
g_args =[]

variables_names_for_ordering ={'f': [], 'g': []}

@njit(cache=True)
def f_ia():
    return ()

@njit(cache=True)
def g_ia():
    return ()

f_jac_args =[]
g_jac_args =[]

jacobian_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}
jacobian_equations ={'dfx': [], 'dfy': [], 'dgx': [], 'dgy': []}