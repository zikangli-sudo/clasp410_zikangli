#!/usr/bin/env python3
'''
Lab 2: Population Control
Andrew Inda

This script models Lab 2, solves Lotka-Volterra equations
for competition and predator-prey systems. Both Euler and
RK8 solvers are used.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use("seaborn-v0_8")


## Derivative functions


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


## LAB QUESTIONS
N = [0.3, 0.6]
Tfinal = 100

# Competition
tE_c, N1E_c, N2E_c = euler_solve(dNdt_comp, *N, dT=1.0, t_final=Tfinal)
tR_c, N1R_c, N2R_c = solve_rk8(dNdt_comp, *N, t_final=Tfinal)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.plot(tE_c, N1E_c, label=r'$N_1$ Euler', lw=2, color='blue')
ax.plot(tE_c, N2E_c, label=r'$N_2$ Euler', lw=2, color='red')
ax.plot(tR_c, N1R_c, linestyle=':', lw=3, color='blue', label=r'$N_1$ RK8')
ax.plot(tR_c, N2R_c, linestyle=':', lw=3, color='red', label=r'$N_2$ RK8')
ax.set_title("Lotka-Volterra Competition Model")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Population/Carrying Cap.")
ax.legend(loc='upper left', frameon=True)
ax.text(0.98, -0.12, "Coefficients: a=1, b=2, c=1, d=3",
        ha='right', va='top', transform=ax.transAxes)


# Predator–Prey
tE_p, N1E_p, N2E_p = euler_solve(dNdt_predprey, *N, dT=0.05, t_final=Tfinal)
tR_p, N1R_p, N2R_p = solve_rk8(dNdt_predprey, *N, t_final=Tfinal)

# Plot
ax = axes[1]
ax.plot(tE_p, N1E_p, label=r'$N_1$ (Prey) Euler', lw=2, color='blue')
ax.plot(tE_p, N2E_p, label=r'$N_2$ (Predator) Euler', lw=2, color='red')
ax.plot(tR_p, N1R_p, linestyle=':', lw=3, color='blue', label=r"$N_1$ (Prey) RK8")
ax.plot(tR_p, N2R_p, linestyle=':', lw=3, color='red', label=r"$N_2$ (Predator) RK8")
ax.set_title("Lotka-Volterra Predator-Prey Model")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Population/Carrying Cap.")
ax.legend(loc='upper left', frameon=True)

fig.tight_layout()

# plt.show()