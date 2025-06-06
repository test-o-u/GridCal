from numba import njit
from numpy import *

@njit(cache=True)
def f_update():
    return zeros((0, 0))

@njit(cache=True)
def g_update(P_end, P_origin, Q_end, Q_origin, b, bsh, g):
    return array([[-Q_end*Q_origin*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin)) + Q_origin**2*g],
                  [-Q_end*Q_origin*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin)) - Q_origin**2*(b + (1/2)*bsh)],
                  [Q_end**2*g - Q_end*Q_origin*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin))],
                  [-Q_end**2*(b + (1/2)*bsh) - Q_end*Q_origin*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin))]])

f_args =[]
g_args =['P_end', 'P_origin', 'Q_end', 'Q_origin', 'b', 'bsh', 'g']

variables_names_for_ordering ={'f': [], 'g': ['P_origin', 'Q_origin', 'P_end', 'Q_end']}

@njit(cache=True)
def f_ia():
    return ()

@njit(cache=True)
def g_ia(P_end, P_origin, Q_end, Q_origin, b, bsh, g):
    return (-Q_end*Q_origin*(b*cos(P_end - P_origin) + g*sin(P_end - P_origin)), -Q_end*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin)) + 2*Q_origin*g, -Q_end*Q_origin*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin)), -Q_origin*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin)), -Q_end*Q_origin*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin)), -Q_end*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin)) - 2*Q_origin*(b + (1/2)*bsh), -Q_end*Q_origin*(b*sin(P_end - P_origin) - g*cos(P_end - P_origin)), -Q_origin*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin)), -Q_end*Q_origin*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin)), -Q_end*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin)), -Q_end*Q_origin*(b*cos(P_end - P_origin) - g*sin(P_end - P_origin)), 2*Q_end*g - Q_origin*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin)), -Q_end*Q_origin*(-b*sin(P_end - P_origin) - g*cos(P_end - P_origin)), -Q_end*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin)), -Q_end*Q_origin*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin)), -2*Q_end*(b + (1/2)*bsh) - Q_origin*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin)),)

f_jac_args =[]
g_jac_args =['P_end', 'P_origin', 'Q_end', 'Q_origin', 'b', 'bsh', 'g']

jacobian_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]}
jacobian_equations ={'dfx': [],
 'dfy': [],
 'dgx': [],
 'dgy': ['-Q_end*Q_origin*(b*cos(P_end - P_origin) + g*sin(P_end - P_origin))',
         '-Q_end*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin)) + 2*Q_origin*g',
         '-Q_end*Q_origin*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin))',
         '-Q_origin*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin))',
         '-Q_end*Q_origin*(-b*sin(P_end - P_origin) + g*cos(P_end - P_origin))',
         '-Q_end*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin)) - 2*Q_origin*(b + bsh/2)',
         '-Q_end*Q_origin*(b*sin(P_end - P_origin) - g*cos(P_end - P_origin))',
         '-Q_origin*(-b*cos(P_end - P_origin) - g*sin(P_end - P_origin))',
         '-Q_end*Q_origin*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin))',
         '-Q_end*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin))',
         '-Q_end*Q_origin*(b*cos(P_end - P_origin) - g*sin(P_end - P_origin))',
         '2*Q_end*g - Q_origin*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin))',
         '-Q_end*Q_origin*(-b*sin(P_end - P_origin) - g*cos(P_end - P_origin))',
         '-Q_end*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin))',
         '-Q_end*Q_origin*(b*sin(P_end - P_origin) + g*cos(P_end - P_origin))',
         '-2*Q_end*(b + bsh/2) - Q_origin*(-b*cos(P_end - P_origin) + g*sin(P_end - P_origin))']}