#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lab 02 — Population Control: Lotka-Volterra (Competition & Predator-Prey)

What this script does
---------------------
• Implements two ODE solvers:
    - Forward Euler (fixed step)
    - DOP853 (adaptive RK8 via scipy.integrate.solve_ivp)
• Provides models:
    - Competition (normalized carrying capacity)
    - Predator-Prey (classic Lotka-Volterra)
• Reproduces all figures for Q1-Q3:
    Q1: Solver comparison + time-step experiments
    Q2: Competition outcomes & IC sweeps (coexistence/exclusion/extinction)
    Q3: Predator-Prey time series + phase diagrams with nullclines

How to run everything
---------------------
$ python3 lab02.py
    - Generates the full set of figures for Q1-Q3 in sequence.

-----------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    """
    Competition model ODEs. Two species competing.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Logistic growth - competition terms
    dN1dt = a * N1 * (1 - N1) - b * N1 * N2
    dN2dt = c * N2 * (1 - N2) - d * N1 * N2
    return [dN1dt, dN2dt]


def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    """
    Predator-prey model ODEs. Prey growth and predator hunting them.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Prey grows, eaten by predator
    dN1dt = a * N1 - b * N1 * N2
    # Predator dies, grows when eating prey
    dN2dt = -c * N2 + d * N1 * N2
    return [dN1dt, dN2dt]


## Euler solver
def euler_solve(func, N1_init=0.3, N2_init=0.6, dT=0.1, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    Euler solver (fixed step). Models the populations step by step.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Build time array
    time = np.arange(0, t_final + dT, dT)
    # Storage arrays
    N1 = np.zeros_like(time)
    N2 = np.zeros_like(time)
    # Set initial values
    N1[0], N2[0] = N1_init, N2_init

    # March forward in time
    for i in range(1, len(time)):
        dN1dt, dN2dt = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        # Euler update: new = old + step * slope
        N1[i] = N1[i-1] + dT * dN1dt
        N2[i] = N2[i-1] + dT * dN2dt
    return time, N1, N2


## RK8 solver
def solve_rk8(func, N1_init=0.3, N2_init=0.6, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    RK8 solver (DOP853, adaptive step). Models smoother populations and takes smaller steps when necessary.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : max step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Call SciPy ODE solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=(a, b, c, d), method="DOP853", max_step=dT)
    return result.t, result.y[0], result.y[1]


# ====== Plots for Problem 1 ======
def plot_competition():
    """
    Competition model: Euler as solid lines, RK8 as dashed lines.
    """
    a, b, c, d = 1, 2, 1, 3
    N1_0, N2_0 = 0.3, 0.6
    T = 100.0

    # Euler: step size 1 year
    t_e, N1_e, N2_e = euler_solve(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=1.0, t_final=T, a=a, b=b, c=c, d=d
    )

    # RK8: max_step = 1 for a fair comparison
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=1.0, t_final=T, a=a, b=b, c=c, d=d
    )

    plt.figure(figsize=(8, 5))
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=1)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=1)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=1)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=1)')

    plt.title('Competition model: Euler (solid) vs RK8 (dashed)')
    plt.xlabel('Time (years)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_competition_addition():
    """
    Competition model: Euler as solid lines, RK8 as dashed lines.
    """
    a, b, c, d = 1, 2, 1, 3
    N1_0, N2_0 = 0.3, 0.6
    T = 100.0

    # Base set: dt = 1.0
    t_e, N1_e, N2_e = euler_solve(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=1.0, t_final=T, a=a, b=b, c=c, d=d
    )
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=1.0, t_final=T, a=a, b=b, c=c, d=d
    )

    plt.figure(figsize=(8, 5))
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=1)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=1)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=1)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=1)')

    # Added: dt = 0.1
    t_e, N1_e, N2_e = euler_solve(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=0.1, t_final=T, a=a, b=b, c=c, d=d
    )
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=0.1, t_final=T, a=a, b=b, c=c, d=d
    )
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=0.1)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=0.1)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=0.1)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=0.1)')

    # Added: dt = 0.01
    t_e, N1_e, N2_e = euler_solve(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=0.01, t_final=T, a=a, b=b, c=c, d=d
    )
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=0.01, t_final=T, a=a, b=b, c=c, d=d
    )
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=0.01)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=0.01)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=0.01)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=0.01)')

    plt.title('Competition model: Euler (solid) vs RK8 (dashed)', fontsize=18)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Population', fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def plot_predprey():
    """
    Predator-Prey model: Euler as solid lines, RK8 as dashed lines.
    """
    a, b, c, d = 1, 2, 1, 3
    N1_0, N2_0 = 0.3, 0.6
    T = 100.0

    # Euler: step size 0.05 year
    t_e, N1_e, N2_e = euler_solve(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.05, t_final=T, a=a, b=b, c=c, d=d
    )

    # RK8: max_step = 0.05 for a fair comparison
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.05, t_final=T, a=a, b=b, c=c, d=d
    )

    plt.figure(figsize=(8, 5))
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=0.05)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=0.05)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=0.05)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=0.05)')

    plt.title('Predator–Prey: Euler (solid) vs RK8 (dashed)')
    plt.xlabel('Time (years)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predprey_addition():
    """
    Predator-Prey model: Euler as solid lines, RK8 as dashed lines.
    """
    a, b, c, d = 1, 2, 1, 3
    N1_0, N2_0 = 0.3, 0.6
    T = 100.0

    # Base set: dt = 0.05
    t_e, N1_e, N2_e = euler_solve(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.05, t_final=T, a=a, b=b, c=c, d=d
    )
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.05, t_final=T, a=a, b=b, c=c, d=d
    )

    plt.figure(figsize=(8, 5))
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=0.05)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=0.05)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=0.05)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=0.05)')

    t_e, N1_e, N2_e = euler_solve(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.03, t_final=T, a=a, b=b, c=c, d=d
    )
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.03, t_final=T, a=a, b=b, c=c, d=d
    )
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=0.03)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=0.03)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=0.03)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=0.03)')

    t_e, N1_e, N2_e = euler_solve(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.02, t_final=T, a=a, b=b, c=c, d=d
    )
    t_rk, N1_rk, N2_rk = solve_rk8(
        dNdt_predprey, N1_init=N1_0, N2_init=N2_0, dT=0.02, t_final=T, a=a, b=b, c=c, d=d
    )
    plt.plot(t_e, N1_e, linewidth=2, label='Euler N1 (dT=0.02)')
    plt.plot(t_e, N2_e, linewidth=2, label='Euler N2 (dT=0.02)')
    plt.plot(t_rk, N1_rk, linestyle='--', label='RK8 N1 (max_step=0.02)')
    plt.plot(t_rk, N2_rk, linestyle='--', label='RK8 N2 (max_step=0.02)')

    plt.title('Predator–Prey: Euler (solid) vs RK8 (dashed)')
    plt.xlabel('Time (years)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========= Helpers for competition model (Q2) =========
def coexistence_point(a, b, c, d):
    """
    Return interior coexistence equilibrium (if it exists and is positive) for K1=K2=1.
    Uses alpha_12 = b/a and alpha_21 = d/c. Returns (N1*, N2*) or None.
    """
    alpha12 = b / a
    alpha21 = d / c
    denom = 1.0 - alpha12 * alpha21
    if denom == 0:
        return None
    N1_star = (1.0 - alpha12) / denom
    N2_star = (1.0 - alpha21) / denom
    if (N1_star > 0) and (N2_star > 0):
        return (N1_star, N2_star)
    return None


def _plot_case_on_ax(ax, title, N0, coeffs, T=100.0, dt=0.02):
    """
    Plot one competition case on a given Axes:
      - Euler (solid), step=dt
      - DOP853 (dashed), max_step=dt
      - If a positive interior coexistence exists, draw thin dotted reference lines
    """
    a, b, c, d = coeffs
    N1_0, N2_0 = N0

    # Euler
    te, N1e, N2e = euler_solve(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=dt, t_final=T, a=a, b=b, c=c, d=d
    )
    # RK8
    trk, N1rk, N2rk = solve_rk8(
        dNdt_comp, N1_init=N1_0, N2_init=N2_0, dT=dt, t_final=T, a=a, b=b, c=c, d=d
    )

    # Draw
    ax.plot(te, N1e, linewidth=2, label=f"Euler N1 (dt={dt})")
    ax.plot(te, N2e, linewidth=2, label=f"Euler N2 (dt={dt})")
    ax.plot(trk, N1rk, linestyle="--", label=f"RK8 N1 (max_step={dt})")
    ax.plot(trk, N2rk, linestyle="--", label=f"RK8 N2 (max_step={dt})")

    # Interior coexistence if it exists
    eq = coexistence_point(a, b, c, d)
    if eq is not None:
        ax.axhline(eq[0], linestyle=":", linewidth=1)
        ax.axhline(eq[1], linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Population")
    ax.grid(True, alpha=0.3)


def plot_case_single(title, N0, coeffs, T=100.0, dt=0.02):
    """
    Standalone figure for a single parameter + IC case (Euler solid vs RK8 dashed).
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _plot_case_on_ax(ax, title=title, N0=N0, coeffs=coeffs, T=T, dt=dt)
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show()


# ========= Four canonical outcomes for Q2 =========
def fig0_both_extinct():
    """
    Both go extinct: choose negative intrinsic growth (a<0, c<0),
    while keeping cross-competition positive.
    This makes (N1, N2) -> (0, 0) a stable outcome.
    """
    plot_case_single(
        title="Figure 0 — Both species go extinct",
        N0=(0.4, 0.5),
        coeffs=(-0.2, 0.6, -0.2, 0.7),  # a<0, c<0 -> decay to (0,0)
        T=100.0,
        dt=0.02,
    )


def fig1_coexistence():
    """
    Coexistence equilibrium: alpha12=b/a<1 and alpha21=d/c<1.
    """
    plot_case_single(
        title="Figure 1 — Coexistence (stable interior equilibrium)",
        N0=(0.3, 0.6),
        coeffs=(1.0, 0.5, 1.0, 0.6),  # a,b,c,d  => alpha12=0.5, alpha21=0.6
        T=100.0,
        dt=0.02,
    )


def fig2_species1_wins():
    """
    Species 1 wins: alpha12<1<alpha21, so (N1,N2)->(1,0).
    """
    plot_case_single(
        title="Figure 2 — Species 1 wins",
        N0=(0.2, 0.6),
        coeffs=(1.0, 0.6, 1.0, 1.4),
        T=100.0,
        dt=0.02,
    )


def fig3_species2_wins():
    """
    Species 2 wins: alpha21<1<alpha12, so (N1,N2)->(0,1).
    """
    plot_case_single(
        title="Figure 3 — Species 2 wins",
        N0=(0.6, 0.2),
        coeffs=(1.0, 1.5, 1.0, 0.5),
        T=100.0,
        dt=0.02,
    )


def fig_all_2x2_summary():
    """
    One 2x2 summary panel: [both extinct, coexistence; species1 wins, species2 wins].
    This matches the four algebraic equilibria classes and shows which are realized
    under the chosen coefficients and initial conditions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

    _plot_case_on_ax(
        axes[0, 0],
        "Both extinct (a<0, c<0)",
        N0=(0.4, 0.5),
        coeffs=(-0.2, 0.6, -0.2, 0.7),
        T=100.0,
        dt=0.02,
    )
    axes[0, 0].legend()

    _plot_case_on_ax(
        axes[0, 1],
        "Coexistence (alpha12<1, alpha21<1)",
        N0=(0.3, 0.6),
        coeffs=(1.0, 0.5, 1.0, 0.6),
        T=100.0,
        dt=0.02,
    )
    axes[0, 1].legend()

    _plot_case_on_ax(
        axes[1, 0],
        "Species 1 wins (alpha12<1<alpha21)",
        N0=(0.2, 0.6),
        coeffs=(1.0, 0.6, 1.0, 1.4),
        T=100.0,
        dt=0.02,
    )
    axes[1, 0].legend()

    _plot_case_on_ax(
        axes[1, 1],
        "Species 2 wins (alpha21<1<alpha12)",
        N0=(0.6, 0.2),
        coeffs=(1.0, 1.5, 1.0, 0.5),
        T=100.0,
        dt=0.02,
    )
    axes[1, 1].legend()

    fig.suptitle("Q2 summary — competition model outcomes via ICs and coefficients")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ========= IC-sweep figures to show initial-condition effects =========
def fig_ic_effect_coexistence():
    """
    Initial-condition sweep under coexistence (alpha12<1, alpha21<1):
    different ICs -> same interior equilibrium (transients differ).
    """
    coeffs = (1.0, 0.5, 1.0, 0.6)  # alpha12=0.5, alpha21=0.6
    T, dt = 100.0, 0.02
    ICs = [(0.3, 0.6), (0.1, 0.9), (0.8, 0.2), (0.55, 0.55)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (n10, n20) in ICs:
        te, N1e, N2e = euler_solve(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        trk, N1rk, N2rk = solve_rk8(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        ax.plot(te, N1e, lw=1.8, label=f"Euler N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(te, N2e, lw=1.8, label=f"Euler N2, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N1rk, ls="--", lw=1.2, label=f"RK8 N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N2rk, ls="--", lw=1.2, label=f"RK8 N2, IC=({n10:.2f},{n20:.2f})")

    eq = coexistence_point(*coeffs)
    if eq is not None:
        ax.axhline(eq[0], ls=":", lw=1.0)
        ax.axhline(eq[1], ls=":", lw=1.0)

    ax.set_title("IC sweep — coexistence (alpha12<1, alpha21<1)", fontsize=18)
    ax.set_xlabel("Time (years)", fontsize=14)
    ax.set_ylabel("Population",fontsize=14)
    ax.grid(True, alpha=0.3)
    # moved legend farther down and added extra bottom margin
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.subplots_adjust(bottom=0.32)
    plt.show()


def fig_ic_effect_exclusion_species1_wins():
    """
    IC sweep under competitive exclusion (species 1 wins):
    alpha12<1<alpha21. Different ICs -> same winner (N1->1, N2->0).
    """
    coeffs = (1.0, 0.6, 1.0, 1.4)  # alpha12=0.6<1<alpha21=1.4
    T, dt = 100.0, 0.02
    ICs = [(0.2, 0.6), (0.6, 0.6), (0.1, 0.2), (0.9, 0.8)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (n10, n20) in ICs:
        te, N1e, N2e = euler_solve(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        trk, N1rk, N2rk = solve_rk8(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        ax.plot(te, N1e, lw=1.8, label=f"Euler N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(te, N2e, lw=1.8, label=f"Euler N2, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N1rk, ls="--", lw=1.2, label=f"RK8 N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N2rk, ls="--", lw=1.2, label=f"RK8 N2, IC=({n10:.2f},{n20:.2f})")

    ax.set_title("IC sweep — exclusion (alpha12<1<alpha21): species 1 wins", fontsize=18)
    ax.set_xlabel("Time (years)", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.25))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.subplots_adjust(bottom=0.32)
    plt.show()


def fig_ic_effect_exclusion_species2_wins():
    """
    IC sweep under competitive exclusion (species 2 wins):
    alpha21<1<alpha12. Different ICs -> same winner (N1->0, N2->1).
    """
    coeffs = (1.0, 1.5, 1.0, 0.5)  # alpha21=0.5<1<alpha12=1.5
    T, dt = 100.0, 0.02
    ICs = [(0.6, 0.2), (0.9, 0.1), (0.4, 0.7), (0.2, 0.9)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (n10, n20) in ICs:
        te, N1e, N2e = euler_solve(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        trk, N1rk, N2rk = solve_rk8(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        ax.plot(te, N1e, lw=1.8, label=f"Euler N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(te, N2e, lw=1.8, label=f"Euler N2, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N1rk, ls="--", lw=1.2, label=f"RK8 N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N2rk, ls="--", lw=1.2, label=f"RK8 N2, IC=({n10:.2f},{n20:.2f})")

    ax.set_title("IC sweep — exclusion (alpha21<1<alpha12): species 2 wins", fontsize=18)
    ax.set_xlabel("Time (years)", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.25))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.subplots_adjust(bottom=0.32)
    plt.show()


def fig_ic_effect_both_extinct():
    """
    IC sweep when both go extinct:
    choose a<0 and c<0 (negative intrinsic growth), cross-competition positive.
    Different ICs -> all decay to (0,0); ICs change only the decay path/time.
    """
    coeffs = (-0.2, 0.6, -0.2, 0.7)  # a<0, c<0
    T, dt = 100.0, 0.02
    ICs = [(0.4, 0.5), (0.9, 0.9), (0.1, 0.7), (0.7, 0.1)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (n10, n20) in ICs:
        te, N1e, N2e = euler_solve(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        trk, N1rk, N2rk = solve_rk8(
            dNdt_comp, N1_init=n10, N2_init=n20, dT=dt, t_final=T,
            a=coeffs[0], b=coeffs[1], c=coeffs[2], d=coeffs[3]
        )
        ax.plot(te, N1e, lw=1.8, label=f"Euler N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(te, N2e, lw=1.8, label=f"Euler N2, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N1rk, ls="--", lw=1.2, label=f"RK8 N1, IC=({n10:.2f},{n20:.2f})")
        ax.plot(trk, N2rk, ls="--", lw=1.2, label=f"RK8 N2, IC=({n10:.2f},{n20:.2f})")

    ax.set_title("IC sweep — both go extinct (a<0, c<0)", fontsize=18)
    ax.set_xlabel("Time (years)", fontsize=14)
    ax.set_ylabel("Population", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.25))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.subplots_adjust(bottom=0.32)
    plt.show()

# ========= Helpers for Question 3 (predator–prey only) =========
def predprey_equilibrium(a, b, c, d):
    """
    Return the interior equilibrium for classic Lotka-Volterra:
        N1* = c/d (prey),  N2* = a/b (predator).
    """
    return (c / d, a / b)


def simulate_pp(IC, coeffs, T=100.0, dt=0.02, use_euler=False):
    """
    Simulate predator-prey with given initial conditions and coefficients.
    Returns time, N1, N2 arrays.
    """
    a, b, c, d = coeffs
    N1_0, N2_0 = IC
    if use_euler:
        return euler_solve(dNdt_predprey, N1_init=N1_0, N2_init=N2_0,
                           dT=dt, t_final=T, a=a, b=b, c=c, d=d)
    else:
        return solve_rk8(dNdt_predprey, N1_init=N1_0, N2_init=N2_0,
                         dT=dt, t_final=T, a=a, b=b, c=c, d=d)


def plot_timeseries(ax, t, N1, N2, label_prefix=""):
    """Plot N1(t) and N2(t) on a single axes."""
    ax.plot(t, N1, linewidth=2, label=f'{label_prefix} Prey N1')
    ax.plot(t, N2, linewidth=2, label=f'{label_prefix} Predator N2')
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Population")
    ax.grid(True, alpha=0.3)


def plot_phase(ax, t, N1, N2, coeffs, label_prefix="", draw_field=True):
    """
    Plot phase trajectory (N1 on x, N2 on y), with nullclines and equilibrium.
    Optionally overlay a sparse vector field to show direction of motion.
    """
    a, b, c, d = coeffs
    # Trajectory
    ax.plot(N1, N2, linewidth=2, label=f'{label_prefix} trajectory')

    # Equilibrium and nullclines
    n1_star, n2_star = predprey_equilibrium(a, b, c, d)
    ax.scatter([n1_star], [n2_star], marker='o', s=40, zorder=5, label='equilibrium')
    # Nullclines: dN1/dt=0 -> N2=a/b (horizontal); dN2/dt=0 -> N1=c/d (vertical)
    ax.axhline(n2_star, linestyle=':', linewidth=1)
    ax.axvline(n1_star, linestyle=':', linewidth=1)

    # Optional direction field (sparse)
    if draw_field:
        x = np.linspace(max(1e-3, 0.1*n1_star), 2.5*n1_star, 15)
        y = np.linspace(max(1e-3, 0.1*n2_star), 2.5*n2_star, 15)
        X, Y = np.meshgrid(x, y)
        U = a*X - b*X*Y
        V = -c*Y + d*X*Y
        # Normalize arrows to avoid clutter
        speed = np.sqrt(U**2 + V**2) + 1e-12
        U, V = U/speed, V/speed
        ax.quiver(X, Y, U, V, angles='xy', width=0.003, scale=25)

    ax.set_xlabel("Prey N1")
    ax.set_ylabel("Predator N2")
    ax.grid(True, alpha=0.3)


# ========= Scenarios (each shows time series + phase diagram) =========
def fig1_change_initial_conditions():
    """
    Scenario 1 — Same coefficients, different initial conditions.
    Regular closed orbits at different radii around the same equilibrium.
    """
    a, b, c, d = 1.0, 2.0, 1.0, 3.0          # baseline coefficients
    coeffs = (a, b, c, d)
    ICs = [(0.3, 0.6), (0.6, 0.3), (0.9, 0.4)]  # vary initial conditions
    colors = ["C0", "C1", "C2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (ic, col) in zip(ICs, colors):
        t, N1, N2 = simulate_pp(ic, coeffs, T=100.0, dt=0.02, use_euler=False)
        ax1.plot(t, N1, color=col, linewidth=2, label=f"N1, IC={ic}")
        ax1.plot(t, N2, color=col, linewidth=2, linestyle='--', label=f"N2, IC={ic}")
        plot_phase(ax2, t, N1, N2, coeffs, label_prefix=f"IC={ic}", draw_field=False)

    ax1.set_title("Time series — same coefficients, different ICs")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Population")
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2)

    ax2.set_title("Phase diagram — closed orbits at different radii")
    ax2.legend()
    fig.suptitle("Predator–Prey: effect of initial conditions (regular closed cycles)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def fig2_stronger_predation():
    """
    Scenario 2 — Increase predation rate (b↑).
    Equilibrium predator level N2* = a/b decreases; cycles skew with larger predator peaks.
    """
    coeffs = (1.0, 3.0, 1.0, 3.0)  # b up to 3.0
    IC = (0.5, 0.4)

    t, N1, N2 = simulate_pp(IC, coeffs, T=100.0, dt=0.02, use_euler=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_timeseries(ax1, t, N1, N2, label_prefix="strong predation")
    ax1.set_title("Time series — stronger predation (b=3)")

    plot_phase(ax2, t, N1, N2, coeffs, label_prefix="b=3", draw_field=True)
    ax2.set_title("Phase diagram — center shifts (N2* = a/b smaller)")
    fig.suptitle("Predator–Prey: effect of predation rate b")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def fig3_better_conversion_lower_decay():
    """
    Scenario 3 — Predator survives more easily (c↓) and converts prey better (d↑).
    Equilibrium shifts left (N1* = c/d smaller), predator oscillations become broader.
    """
    coeffs = (1.0, 2.0, 0.5, 2.0)  # c down to 0.5, d up to 2.0
    IC = (0.4, 0.3)

    t, N1, N2 = simulate_pp(IC, coeffs, T=100.0, dt=0.02, use_euler=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_timeseries(ax1, t, N1, N2, label_prefix="c=0.5, d=2.0")
    ax1.set_title("Time series — predators persist more easily")

    plot_phase(ax2, t, N1, N2, coeffs, label_prefix="c=0.5, d=2.0", draw_field=True)
    ax2.set_title("Phase diagram — N1* = c/d becomes smaller")
    fig.suptitle("Predator–Prey: effect of c (decay) and d (conversion)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

def fig2_weaker_predation():
    """
    Opposite of 'stronger predation': decrease b.
    Result: N2* = a/b becomes larger -> horizontal nullcline N2=a/b moves UP,
    the phase-diagram center moves UP, predator mean increases.
    """
    coeffs = (1.0, 1.2, 1.0, 3.0)   # smaller b than baseline (e.g., b from 2 -> 1.2)
    IC = (0.5, 0.4)

    t, N1, N2 = simulate_pp(IC, coeffs, T=100.0, dt=0.02, use_euler=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_timeseries(ax1, t, N1, N2, label_prefix="weaker predation")
    ax1.set_title("Time series — weaker predation (b = 1.2)")

    plot_phase(ax2, t, N1, N2, coeffs, label_prefix="b = 1.2", draw_field=True)
    ax2.set_title("Phase diagram — center moves UP (N2* = a/b larger)")
    fig.suptitle("Predator–Prey: effect of smaller predation rate b")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

def fig3_harder_persistence_high_c_low_d():
    """
    Opposite of 'lower c and higher d': increase c and decrease d.
    Result: N1* = c/d becomes larger -> vertical nullcline N1=c/d moves RIGHT,
    the phase-diagram center moves RIGHT; predators need higher prey to persist.
    """
    coeffs = (1.0, 2.0, 1.6, 1.2)   # larger c, smaller d (e.g., c=1.6, d=1.2)
    IC = (0.4, 0.3)

    t, N1, N2 = simulate_pp(IC, coeffs, T=100.0, dt=0.02, use_euler=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_timeseries(ax1, t, N1, N2, label_prefix="c = 1.6, d = 1.2")
    ax1.set_title("Time series — higher decay & lower conversion")

    plot_phase(ax2, t, N1, N2, coeffs, label_prefix="c = 1.6, d = 1.2", draw_field=True)
    ax2.set_title("Phase diagram — center moves RIGHT (N1* = c/d larger)")
    fig.suptitle("Predator–Prey: effect of larger c and smaller d")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

if __name__ == "__main__":
    # Q1
    plot_competition()
    plot_competition_addition()
    plot_predprey()
    plot_predprey_addition()

    # Q2
    fig_all_2x2_summary()

    fig_ic_effect_coexistence()
    fig_ic_effect_exclusion_species1_wins()
    fig_ic_effect_exclusion_species2_wins()
    fig_ic_effect_both_extinct()
    
    # Q3
    fig1_change_initial_conditions()
    fig2_stronger_predation()
    fig3_better_conversion_lower_decay()
    fig2_weaker_predation()
    fig3_harder_persistence_high_c_low_d()
