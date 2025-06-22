# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np
from GridCalEngine.Utils.Symbolic.symbolic import Var, Expr, Const
from GridCalEngine.Utils.Symbolic.block import Block, integrator, gain, adder, compose_block, constant, pi_controller
from GridCalEngine.Utils.Symbolic.engine import BlockSystem




def demo_oscillator():
    x, blk_x = integrator(None, "x")
    v, blk_v = integrator(None, "v")

    blk_x.state_eqs[0] = v
    blk_v.state_eqs[0] = Const(-1) * x

    sys = BlockSystem([blk_x, blk_v])
    x0 = np.array([1.0, 0.0])

    t_rk4, y_rk4 = sys.simulate(0, 10, 0.01, x0, method="rk4")
    t_eu, y_eu   = sys.simulate(0, 10, 0.01, x0, method="euler")

    plt.figure("Harmonic oscillator")
    plt.plot(t_rk4, y_rk4[:, 0], label="x (RK4)")
    plt.plot(t_eu,  y_eu[:, 0], label="x (Euler)")
    plt.plot(t_rk4, y_rk4[:, 1], "--", label="v (RK4)")
    plt.xlabel("time [s]"); plt.ylabel("state"); plt.legend()

# --------------------------------------------------------------------------------------
# Demo 2 – 3‑bus power system (swing + DC PF)
# --------------------------------------------------------------------------------------

def demo_power_system():
    M, D, Pm = 5.0, 0.1, 1.0
    P2, P3 = 0.8, 0.6
    B12, B13, B23 = 5.0, 3.0, 4.0

    # State vars
    delta, blk_delta = integrator(None, "delta")
    omega, blk_omega = integrator(None, "omega")

    blk_delta.state_eqs[0] = omega  # δ̇ = ω

    # Algebraic bus angles θ2, θ3
    theta2 = Var("theta2"); theta3 = Var("theta3")
    a11, a12, a21, a22 = B12 + B23, -B23, -B23, B13 + B23
    det = a11 * a22 - a12 * a21
    theta2_expr = (a22 * Const(P2) - a12 * Const(P3)) / Const(det) + (a22 * B12 - a12 * B13) / Const(det) * delta
    theta3_expr = (-a21 * Const(P2) + a11 * Const(P3)) / Const(det) + (-a21 * B12 + a11 * B13) / Const(det) * delta

    blk_angles = Block([theta2, theta3], [theta2 - theta2_expr, theta3 - theta3_expr], [], [], "angles")

    Pe = Const(B12) * (delta - theta2) + Const(B13) * (delta - theta3)
    blk_omega.state_eqs[0] = (Const(Pm) - Pe - Const(D) * omega) / Const(M)

    sys = BlockSystem([blk_delta, blk_omega, blk_angles])
    x0 = sys.build_init_vector({delta: 0.0, omega: 0.0})

    t, y = sys.simulate(0, 20, 0.01, x0, method="rk4")

    plt.figure("Swing generator")
    plt.plot(t, y[:, 0], label="delta [rad]")
    plt.plot(t, y[:, 1], label="omega [pu]")
    plt.xlabel("time [s]"); plt.legend();

# --------------------------------------------------------------------------------------
# Demo 3 – PI‑controlled first‑order plant
# --------------------------------------------------------------------------------------

def demo_pi_controller():
    # Plant state
    y, blk_y = integrator(None, "y")

    # Reference
    r, blk_r = constant(1.0, "r")

    # Error
    e, blk_err = adder([r, Const(-1) * y], "e")

    # PI controller
    u_ctrl, pi_blocks = pi_controller(e, kp=2.0, ki=1.0)

    # Plant dynamics ẏ = -y + u
    blk_y.state_eqs[0] = Const(-1) * y + u_ctrl

    sys = BlockSystem([blk_r, blk_err, blk_y] + pi_blocks)
    x0 = sys.build_init_vector({y: 0.0})

    t, y_hist = sys.simulate(0, 10, 0.02, x0, method="implicit_euler")

    plt.figure("PI closed loop")
    plt.plot(t, y_hist[:, 0], label="y(t)")
    plt.plot(t, np.ones_like(t), "--", label="reference")
    plt.xlabel("time [s]"); plt.legend()

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    demo_oscillator()
    demo_power_system()
    demo_pi_controller()
    plt.show()