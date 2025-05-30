# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from sympy import symbols, Eq, solve, nsolve, conjugate, I, log, ln, Abs, im, re, exp
import numpy as np
from scipy.optimize import fsolve, root
from numpy import ones_like, zeros_like, full, array
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select
from numpy import greater_equal, less_equal, greater, less, equal
from numpy import logical_and, logical_or, logical_not
from numpy import real, imag, conj, angle, radians, abs
from numpy import arcsin, arccos, arctan, arctan2
from numpy import log


# Sympy solve approach
#
# delta, omega, id, iq, vd, vq, tm, te, vf, Pe, Qe, psid, psiq, a, v, p0s, q0s, xq, p0, q0, _V, _S, _I, _E, deltac, delta0, Vdq, Idq, Id0, Iq0, Vd0, Vq0, tm0, psid0, psiq0, vf0    = symbols('delta omega id iq vd vq tm te vf Pe Qe psid psiq a v p0s q0s xq p0 q0 V S I E deltac delta0 Vdq Idq Id0 Iq0 Vd0 Vq0 tm0 psid0 psiq0 vf0 ')
# gammap, gammaq, ra = symbols('gammap gammaq ra')
#
# gammap = 1
# gammaq = 1
# ra = 1
#
# eq1 = Eq(delta, delta0)
# eq2 = Eq(omega, 1)
# eq3 = Eq(id, Id0)
# eq4 = Eq(iq, Iq0)
# eq5 = Eq(vd, Vd0)
# eq6 = Eq(vq, Vq0)
# eq7 = Eq(tm, tm0)
# eq8 = Eq(te, tm0)
# eq9 = Eq(vf, vf0)
# eq10 = Eq(Pe, Vd0 * Id0 + Vq0 * Iq0)
# eq11 = Eq(Qe, Vq0 * Id0 + Vd0 * Iq0)
# eq12 = Eq(psid, psid0)
# eq13 = Eq(psiq, psiq0)
# eq14 = Eq(p0, p0s * gammap)
# eq15 = Eq(q0, q0s * gammaq)
# eq16 = Eq(_V, v * exp(I *a))
# eq17 = Eq(_S, p0 - I * q0)
# eq18 = Eq(_I, _S/conjugate(_V))
# eq19 = Eq(_E, _V + _I * (ra + I * xq))
# eq20 = Eq(deltac, log(_E/_E))
# eq21 = Eq(delta0, im(deltac))
# eq22 = Eq(Vdq, _V * exp(I * 0.5 * np.pi - deltac))
# eq23 = Eq(Idq, _I * exp(I * 0.5 * np.pi - deltac))
# eq24 = Eq(Id0, re(Idq))
# eq25 = Eq(Iq0, im(Idq))
# eq26 = Eq(Vd0, re(Vdq))
# eq27 = Eq(Vq0, im(Vdq))
# eq28 = Eq(tm0, (Vq0 + ra * Iq0) * Iq0 + (Vd0 + ra * Id0) * Id0)
# eq29 = Eq(psid0, (ra * Iq0) + Vq0)
# eq30 = Eq(psiq0, (ra * Id0) + Vd0)
# eq31 = Eq(vf0, (Vq0 + ra * Iq0) + xq * Id0)
# eq32 = Eq(p0s, 1)
# eq33 = Eq(q0s, 1)
# eq34 = Eq(xq, 1)
# eq35 = Eq(a, 1)
# eq36 = Eq(v, 1)
#
# solution = solve((eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30, eq31, eq32, eq33, eq34, eq35, eq36), (delta, omega, id, iq, vd, vq, tm, te, vf, Pe, Qe, psid, psiq, a, v, p0s, q0s, xq, p0, q0, _V, _S, _I, _E, deltac, delta0, Vdq, Idq, Id0, Iq0, Vd0, Vq0, tm0, psid0, psiq0, vf0))
# print(solution)
#
# # sympy nsolve approach
#
# eq1 = delta - delta0
# eq2 = omega - 1
# eq3 = id - Id0
# eq4 = iq - Iq0
# eq5 = vd - Vd0
# eq6 = vq - Vq0
# eq7 = tm - tm0
# eq8 = te - tm0
# eq9 = vf - vf0
# eq10 = Pe - Vd0 * Id0 + Vq0 * Iq0
# eq11 = Qe - Vq0 * Id0 + Vd0 * Iq0
# eq12 = psid - psid0
# eq13 = psiq - psiq0
# eq14 = p0 - p0s * gammap
# eq15 = q0 - q0s * gammaq
# eq16 = _V - v * exp(I *a)
# eq17 = _S - p0 - I * q0
# eq18 = _I - _S/conj(_V)
# eq19 = _E - _V + _I * (ra + I * xq)
# eq20 = deltac - log(_E/Abs(_E))
# eq21 = delta0 - im(deltac)
# eq22 = Vdq - - _V * exp(I * 0.5 * np.pi - deltac)
# eq23 = Idq - _I * exp(I * 0.5 * np.pi - deltac)
# eq24 = Id0 - re(Idq)
# eq25 = Iq0 - im(Idq)
# eq26 = Vd0 - re(Vdq)
# eq27 = Vq0 - im(Vdq)
# eq28 = tm0 - (Vq0 + ra * Iq0) * Iq0 + (Vd0 + ra * Id0) * Id0
# eq29 = psid0 - (ra * Iq0) + Vq0
# eq30 = psiq0 - (ra * Id0) + Vd0
# eq31 = vf0 - (Vq0 + ra * Iq0) + xq * Id0
# eq32 = p0s - 1
# eq33 = q0s - 1
# eq34 = xq - 1
# eq35 = a - 1
# eq36 = v - 1
#
#
# initial_guess = np.ones(36)
# init_list = initial_guess.tolist()
#
# solution = nsolve((eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30, eq31, eq32, eq33, eq34, eq35, eq36), (delta, omega, id, iq, vd, vq, tm, te, vf, Pe, Qe, psid, psiq, a, v, p0s, q0s, xq, p0, q0, _V, _S, _I, _E, deltac, delta0, Vdq, Idq, Id0, Iq0, Vd0, Vq0, tm0, psid0, psiq0, vf0), init_list)
# print(solution)

# fsolve approach with ANDES system
#
# def equations(vars):
#     delta, omega, id, iq, vd, vq, tm, te, vf, Pe, Qe, psid, psiq, a, v, p0s, q0s, xq, p0, q0, _V, _S, _I, _E, deltac, delta0, Vdq, Idq, Id0, Iq0, Vd0, Vq0, tm0, psid0, psiq0, vf0 = vars
#     gammap = 1
#     gammaq = 1
#     ra = 1
#     eq1 = delta - delta0
#     eq2 = omega - 1
#     eq3 = id - Id0
#     eq4 = iq - Iq0
#     eq5 = vd - Vd0
#     eq6 = vq - Vq0
#     eq7 = tm - tm0
#     eq8 = te - tm0
#     eq9 = vf - vf0
#     eq10 = Pe - Vd0 * Id0 + Vq0 * Iq0
#     eq11 = Qe - Vq0 * Id0 + Vd0 * Iq0
#     eq12 = psid - psid0
#     eq13 = psiq - psiq0
#     eq14 = p0 - p0s * gammap
#     eq15 = q0 - q0s * gammaq
#     eq16 = _V - v * np.exp(1j *a)
#     eq17 = _S - p0 - 1j * q0
#     eq18 = _I - _S/conj(_V)
#     eq19 = _E - _V + _I * (ra + 1j * xq)
#     eq20 = deltac - log(_E/abs(_E))
#     eq21 = delta0 - imag(deltac)
#     eq22 = Vdq - - _V * np.exp(1j * 0.5 * np.pi - deltac)
#     eq23 = Idq - _I * np.exp(1j* 0.5 * np.pi - deltac)
#     eq24 = Id0 - real(Idq)
#     eq25 = Iq0 - imag(Idq)
#     eq26 = Vd0 - real(Vdq)
#     eq27 = Vq0 - imag(Vdq)
#     eq28 = tm0 - (Vq0 + ra * Iq0) * Iq0 + (Vd0 + ra * Id0) * Id0
#     eq29 = psid0 - (ra * Iq0) + Vq0
#     eq30 = psiq0 - (ra * Id0) + Vd0
#     eq31 = vf0 - (Vq0 + ra * Iq0) + xq * Id0
#     eq32 = p0s - 1
#     eq33 = q0s - 1
#     eq34 = xq - 1
#     eq35 = a - 1
#     eq36 = v - 1
#     return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23, eq24, eq25, eq26, eq27, eq28, eq29, eq30, eq31, eq32, eq33, eq34, eq35, eq36]
#
# initial_guess = np.random.rand(36)
# init_list = initial_guess.tolist()
#
# solution = fsolve(equations, init_list)
# print("Solution to the system:", solution)

# fsolve approach with Milano system

def equations(vars):
    ra = 1
    xq = 1
    xd = 1
    v0 = 1
    a0 = 1
    p0 = 1
    q0 = 1
    delta0, omega0, vd0, vq0, id0, iq0, psid0, psiq0, vf0, te0, tm0 = vars
    eq1 = np.angle(((v0 * exp(1j * a0)) + (ra + 1j * xq) * ((p0 - 1j * q0)/conj(v0 * exp(1j * a0)))))-delta0
    eq2 = sqrt(vd0**2 + vq0**2)-v0
    eq3 = sqrt(id0**2 + iq0**2) - (p0 - 1j * q0)/v0
    eq4 = (v0 * exp(1j * a0)) * exp(-1j * (delta0 - np.pi/2)) - (vd0 + 1j * vq0)
    eq5 = ((p0 - 1j * q0)/conj(v0 * exp(1j * a0))) * exp(-1j * (delta0 - np.pi/2)) - (id0 + 1j * iq0)
    eq6 = (vq0 + ra * iq0) * iq0 + (vd0 + ra * id0) * id0 - te0
    eq7 = vq0 + ra * iq0 + xd * id0 - psiq0
    eq8 = vd0 + ra * id0 - xq * iq0 - psid0
    eq9 = psiq0 + id0 - vf0
    eq10 = omega0 - 1
    eq11 = te0 - tm0
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11]

initial_guess = np.random.rand(11)
init_list = initial_guess.tolist()

solution = root(equations, init_list)
print("Solution to the system:", solution)
