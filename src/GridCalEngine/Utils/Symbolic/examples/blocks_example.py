# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np
from GridCalEngine.Utils.Symbolic.symbolic import Var, Expr, Const
from GridCalEngine.Utils.Symbolic.block import Block, integrator, gain, adder, compose_block, constant
from GridCalEngine.Utils.Symbolic.engine import BlockSystem


# --------------------------------------------------------------------------------------
# Quick self‑test / demo (harmonic oscillator)
# --------------------------------------------------------------------------------------
def _figure(title: str):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig, ax


# =============================================================================
# Demo1 – harmonic oscillator
# =============================================================================

def demo_harmonic(method: str, h: float = 0.01, adaptive: bool = False):
    """Run the harmonic oscillator with the chosen integration *method*."""
    print(f"\n★ Harmonic oscillator using {method}…")
    # x'' + x = 0 →let x₁ = x, x₂ = x'  ⇒  ẋ₁ = x₂,  ẋ₂ = −x₁
    x1, blk_int1 = integrator(None, "x1")  # placeholder derivative (to be set)
    x2, blk_int2 = integrator(None, "x2")

    # Define algebraic derivatives now that states exist
    blk_int1.state_eqs[0] = x2
    blk_int2.state_eqs[0] = Const(-1) * x1

    sys = BlockSystem([blk_int1, blk_int2])
    init = [1.0, 0.0]  # x(0) = 1, x'(0) = 0

    if adaptive:
        t, y = sys.simulate(0.0, 10.0, h, init, method="adaptive")
    else:
        t, y = sys.simulate(0.0, 10.0, h, init, method=method)

    fig, ax = _figure(f"Harmonic oscillator – {method}")
    ax.plot(t, y[:, 0], label="x")
    ax.plot(t, y[:, 1], label="ẋ")
    ax.legend()
    return fig, ax


# =============================================================================
# Demo2 – 3‑bus power‑system
# =============================================================================

def demo_power_system(h: float = 0.005):
    print("\n★ 3‑bus power‑system…")
    # Parameters
    M = 5.0  # inertia [pu·s]
    D = 0.1  # damping [pu]
    P_m = 1.0  # mechanical input [pu]
    P2 = 0.8  # load at bus 2 [pu]
    P3 = 0.6  # load at bus 3 [pu]
    # Line susceptances (DC power flow)
    B12 = 5.0
    B13 = 3.0
    B23 = 4.0

    # ------------------------------------------------------------------
    # Block definitions
    # ------------------------------------------------------------------
    delta, blk_int_delta = integrator(None, "delta")  # rotor angle [rad]
    omega, blk_int_omega = integrator(None, "omega")  # speed deviation [pu]

    # Algebraic bus angles θ2, θ3 (solved from DC PF)
    theta2 = Var("theta2")
    theta3 = Var("theta3")

    # Solve linear system manually (Y_bus reduced) for DC PF:
    #   P2 = B12*(theta2 - delta) + B23*(theta2 - theta3)
    #   P3 = B13*(theta3 - delta) + B23*(theta3 - theta2)
    a11 = B12 + B23
    a12 = -B23
    a21 = -B23
    a22 = B13 + B23
    det = a11 * a22 - a12 * a21
    theta2_expr = (a22 * (Const(P2)) - a12 * (Const(P3))) / Const(det) + (a22 * B12 - a12 * B13) / Const(det) * delta
    theta3_expr = (-a21 * (Const(P2)) + a11 * (Const(P3))) / Const(det) + (-a21 * B12 + a11 * B13) / Const(det) * delta

    blk_angles = Block(
        algebraic_vars=[theta2, theta3],
        algebraic_eqs=[theta2 - theta2_expr, theta3 - theta3_expr],
        state_vars=[],
        state_eqs=[],
    )

    # Electrical air‑gap power P_e = Σ B1k (δ − θ_k)
    P_e = (Const(B12) * (delta - theta2) + Const(B13) * (delta - theta3))

    # Complete state derivatives
    blk_int_delta.state_eqs[0] = omega  # δ̇ = ω
    blk_int_omega.state_eqs[0] = (Const(P_m) - P_e - Const(D) * omega) / Const(M)  # ω̇

    sys = BlockSystem([blk_int_delta, blk_int_omega, blk_angles])

    init = [0.0, 0.0]  # δ=0 rad, ω=0 pu at t=0
    t, y = sys.simulate(0.0, 20.0, h, init, method="rk4")

    fig, ax = _figure("3‑bus power‑system – δ & ω")
    ax.plot(t, y[:, 0], label="δ (rad)")
    ax.plot(t, y[:, 1], label="ω (pu)")
    ax.set_ylabel("Value")
    ax.legend()

    return fig, ax

def build_init_vector(sys: BlockSystem, mapping: dict[Var, float]) -> np.ndarray:
    """Return a NumPy 1‑D vector whose elements align with sys.state_vars."""
    return np.array([mapping.get(v, 0.0) for v in sys.state_vars], dtype=float)


def demo_pi_controller():
    print("★ Demo 3 – PI controller closed loop…")

    # Plant state (output) variable ẏ = u
    y, blk_dummy = integrator(Const(0), "y")

    # Reference 1.0 p.u.
    ref, blk_ref = constant(1.0, "ref")

    # Error signal err = ref − y
    err, blk_err = adder([ref, -y], "err")

    Kp, Ki = 2.0, 1.0

    # PI internals
    u_p, blk_kp = gain(Kp, err, "u_p")
    ie, blk_int = integrator(err, "ie")
    u_i, blk_ki = gain(Ki, ie, "u_i")
    u, blk_sum  = adder([u_p, u_i], "u")

    # First‑order plant ẏ = u
    blk_plant = Block(
        algebraic_vars=[],
        algebraic_eqs=[],
        state_vars=[y],
        state_eqs=[u],
        name="plant",
    )

    sys = BlockSystem([
        blk_ref, blk_err,
        blk_kp, blk_int, blk_ki, blk_sum,
        blk_plant,
    ])

    init_vec = build_init_vector(sys, {y: 0.0, ie: 0.0})

    t, states = sys.simulate(0.0, 10.0, 0.01, init_vec, method="rk4")

    idx_y, idx_ie = map(sys.state_vars.index, (y, ie))

    y_trace  = states[:, idx_y]
    ie_trace = states[:, idx_ie]
    err_trace = 1.0 - y_trace
    u_trace  = Kp * err_trace + Ki * ie_trace

    plt.figure("PI – output vs reference")
    plt.plot(t, y_trace, label="y (output)")
    plt.plot(t, np.ones_like(t), "--", label="reference = 1.0")
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.title("PI closed‑loop response")
    plt.legend()

    plt.figure("PI – control effort")
    plt.plot(t, u_trace, label="u (control)")
    plt.xlabel("Time [s]")
    plt.ylabel("u")
    plt.title("PI control effort vs time")
    plt.legend()


# =============================================================================
# Entry‑point
# =============================================================================
if __name__ == "__main__":
    demo_harmonic("rk4")
    demo_harmonic("euler")
    demo_harmonic("rk4", adaptive=True)  # RKF‑45
    demo_power_system()
    demo_pi_controller()
    plt.show()
