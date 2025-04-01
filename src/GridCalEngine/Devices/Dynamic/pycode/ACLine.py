import numpy

from numpy import *

def f_update():
    return array([])


def g_update(a1, a2, b, bsh, g, v1, v2):
    return array([[-g*v1**2 + v1*v2*(b*sin(a1 - a2) + g*cos(a1 - a2))], [v1**2*(b + (1/2)*bsh) + v1*v2*(-b*cos(a1 - a2) + g*sin(a1 - a2))], [g*v2**2 - v1*v2*(-b*sin(a1 - a2) + g*cos(a1 - a2))], [-v1*v2*(-b*cos(a1 - a2) - g*sin(a1 - a2)) - v2**2*(b + (1/2)*bsh)]])


f_args =[]
g_args =['a1', 'a2', 'v1', 'v2']
def f_ia():
    return ()


def g_ia(a1, a2, b, bsh, g, v1, v2):
    return (v1*v2*(b*cos(a1 - a2) - g*sin(a1 - a2)), -2*g*v1 + v2*(b*sin(a1 - a2) + g*cos(a1 - a2)), v1*v2*(-b*cos(a1 - a2) + g*sin(a1 - a2)), v1*(b*sin(a1 - a2) + g*cos(a1 - a2)), v1*v2*(b*sin(a1 - a2) + g*cos(a1 - a2)), 2*v1*(b + (1/2)*bsh) + v2*(-b*cos(a1 - a2) + g*sin(a1 - a2)), v1*v2*(-b*sin(a1 - a2) - g*cos(a1 - a2)), v1*(-b*cos(a1 - a2) + g*sin(a1 - a2)), -v1*v2*(-b*cos(a1 - a2) - g*sin(a1 - a2)), -v2*(-b*sin(a1 - a2) + g*cos(a1 - a2)), -v1*v2*(b*cos(a1 - a2) + g*sin(a1 - a2)), 2*g*v2 - v1*(-b*sin(a1 - a2) + g*cos(a1 - a2)), -v1*v2*(b*sin(a1 - a2) - g*cos(a1 - a2)), -v2*(-b*cos(a1 - a2) - g*sin(a1 - a2)), -v1*v2*(-b*sin(a1 - a2) + g*cos(a1 - a2)), -v1*(-b*cos(a1 - a2) - g*sin(a1 - a2)) - 2*v2*(b + (1/2)*bsh),)


f_jac_args =[]
g_jac_args =['a1', 'a2', 'b', 'bsh', 'g', 'v1', 'v2']
jacobian_info = {'dfx': [], 'dfy': [], 'dgx': [], 'dgy': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]}