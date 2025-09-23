#!/usr/bin/env python3

'''
This files solves the N-layer atmosphere problem for Lab 01 and all subparts.

'''

import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
sigma = 5.67E-8  #Units: W/m2/K-4

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350, debug=False):
    '''
    docstring
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:
                A[i, j] = -2 + 1 * (j == 0)
            else:
                A[i, j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j - i) - 1)
    if (debug):
        print(A)
                    
    b[0] = -0.25 * s0 * (1-albedo)

    # Invert matrix:
    Ainv = np.linalg.inv(A) 
    # Get solution:
    fluxes = np.matmul(Ainv, b)

    # Turn fluxes into temperatures
    # Return temperatures to caller.
    # VERIFY!
    Ts = (fluxes[0] / sigma) ** 0.25
    if nlayers >= 1 and epsilon > 0:
        T_layers = (fluxes[1:] / (epsilon * sigma)) ** 0.25
        temps = np.concatenate(([Ts], T_layers))
    else:
        temps = np.array([Ts])
    return temps


# ======================= APPEND FROM HERE (teacher part above unchanged) =======================

# --- Q5 helper: nuclear-winter variant (shortwave absorbed by TOP layer) ---
def n_layer_atmos_sw_at_top(nlayers, epsilon=0.5, albedo=0.33, s0=1350.0):
    """
    Same longwave matrix A as the base model, but move the shortwave source term
    from the surface row to the TOP layer row to represent 'nuclear winter'.
    """
    N = int(nlayers)
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                A[i, j] = -2 + 1 * (j == 0)
            else:
                A[i, j] = (epsilon ** (i > 0)) * (1 - epsilon) ** (abs(j - i) - 1)

    # SW deposited at the top layer (index N); surface gets none
    b[N] = -0.25 * s0 * (1 - albedo)

    fluxes = np.linalg.inv(A) @ b
    Ts = (fluxes[0] / sigma) ** 0.25
    if N >= 1 and epsilon > 0:
        T_layers = (fluxes[1:] / (epsilon * sigma)) ** 0.25
        temps = np.concatenate(([Ts], T_layers))
    else:
        temps = np.array([Ts])
    return temps


# -------------- Q3(a): single-layer, sweep emissivity ----------------
def q3_single_layer_emissivity_curve(s0=1350.0, albedo=0.33, target_ts=288.0):
    eps_grid = np.linspace(0.001, 0.999, 500)
    Ts_grid = np.array([n_layer_atmos(1, epsilon=e, albedo=albedo, s0=s0)[0] for e in eps_grid])

    idx_best = int(np.argmin(np.abs(Ts_grid - target_ts)))
    eps_star = float(eps_grid[idx_best])
    Ts_at_star = float(Ts_grid[idx_best])

    plt.figure()
    plt.plot(eps_grid, Ts_grid, lw=2)
    plt.axhline(target_ts, ls="--", label=f"Target {target_ts:.0f} K")
    plt.xlabel("Emissivity ε (N = 1)")
    plt.ylabel("Surface temperature Ts (K)")
    plt.title("Q3(a): Ts vs ε (S0=1350 W/m², α=0.33, N=1)")
    plt.legend()
    plt.show()

    print("[Q3(a)] epsilon* ~ %.3f -> Ts ~ %.2f K" % (eps_star, Ts_at_star))


# -------------- Q3(b): fixed emissivity, vary layers -----------------
def q3_fixed_emissivity_layers_curve(epsilon_fixed=0.255, s0=1350.0, albedo=0.33, target_ts=288.0):
    N_vals = list(range(0, 35))
    Ts_by_N = np.array([n_layer_atmos(N, epsilon=epsilon_fixed, albedo=albedo, s0=s0)[0] for N in N_vals])

    idx_closest = int(np.argmin(np.abs(Ts_by_N - target_ts)))
    N_closest = int(N_vals[idx_closest])
    Ts_N_closest = float(Ts_by_N[idx_closest])

    plt.figure()
    plt.plot(N_vals, Ts_by_N, marker="o")
    plt.axhline(target_ts, ls="--", label=f"Target {target_ts:.0f} K")
    plt.xlabel("Number of layers N  (ε = 0.255)")
    plt.ylabel("Surface temperature Ts (K)")
    plt.title("Q3(b): Ts vs N (S0=1350 W/m², α=0.33)")
    plt.legend()
    plt.show()

    temps_profile = n_layer_atmos(N_closest, epsilon=epsilon_fixed, albedo=albedo, s0=s0)
    alt_km = np.arange(0, N_closest + 1)
    plt.figure()
    plt.plot(temps_profile, alt_km, marker="s")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Altitude (km)")
    plt.title(f"Q3(b): Altitude Profile (ε={epsilon_fixed}, N={N_closest})")
    plt.show()

    # also report the minimal N giving Ts >= target (if different)
    N_ge = next((int(N) for N, Ts in zip(N_vals, Ts_by_N) if Ts >= target_ts), None)
    print("[Q3(b)] closest N = %d (Ts ~ %.2f K); minimal N with Ts ≥ %.0f K: %s"
          % (N_closest, Ts_N_closest, target_ts, str(N_ge)))


# -------------- Q4: Ts vs N curve (ε=1; default is Venus) -----------
def q4_ts_vs_layers_curve(s0=2600.0, albedo=0.75, target_ts=700.0, Nmax=120, title_note="Venus"):
    """
    Plot Ts(N) with ε=1 and mark the target Ts (default 700 K for Venus).
    You can change parameters to (S0=1350, albedo=0.33, target_ts=288) to show Earth.
    """
    epsilon = 1.0
    N_vals = list(range(0, Nmax + 1))
    Ts_by_N = np.array([n_layer_atmos(N, epsilon=epsilon, albedo=albedo, s0=s0)[0] for N in N_vals])

    idx_closest = int(np.argmin(np.abs(Ts_by_N - target_ts)))
    N_closest = int(N_vals[idx_closest])
    Ts_closest = float(Ts_by_N[idx_closest])

    # minimal N with Ts >= target
    N_ge = next((int(N) for N, Ts in zip(N_vals, Ts_by_N) if Ts >= target_ts), None)

    plt.figure()
    plt.plot(N_vals, Ts_by_N, marker="o")
    plt.axhline(target_ts, ls="--", label=f"Target {target_ts:.0f} K")
    plt.xlabel("Number of layers N  (ε = 1)")
    plt.ylabel("Surface temperature Ts (K)")
    plt.title(f"Q4: {title_note} — Ts vs N (S0={s0:.0f} W/m², α={albedo:.2f})")
    plt.legend()
    plt.show()

    print("[Q4] %s (α=%.2f): closest N = %d (Ts ~ %.2f K); minimal N with Ts ≥ %.0f K: %s"
          % (title_note, albedo, N_closest, Ts_closest, target_ts, str(N_ge)))


# -------------- Q5: nuclear winter profile (top SW absorption) -------
def q5_nuclear_winter_profile(nlayers=5, epsilon=0.5, s0=1350.0, albedo=0.30):
    temps = n_layer_atmos_sw_at_top(nlayers, epsilon=epsilon, albedo=albedo, s0=s0)
    alt_km = np.arange(0, nlayers + 1)

    plt.figure()
    plt.plot(temps, alt_km, marker="o")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Altitude (N Layer)")
    plt.title(f"Q5: Nuclear Winter Profile (N={nlayers}, ε={epsilon}, S0={s0}, α={albedo})")
    plt.show()

    print("[Q5] Nuclear winter: surface Ts ~ %.2f K  (%.1f °C)"
          % (temps[0], temps[0] - 273.15))


# ---------------------- Run chosen parts -----------------------------
if __name__ == "__main__":
    # Q3
    q3_single_layer_emissivity_curve(s0=1350.0, albedo=0.33, target_ts=288.0)
    q3_fixed_emissivity_layers_curve(epsilon_fixed=0.255, s0=1350.0, albedo=0.33, target_ts=288.0)

    # Q4
    q4_ts_vs_layers_curve(s0=2600.0, albedo=0.75, target_ts=700.0, Nmax=120, title_note="Venus")

    # Q5
    q5_nuclear_winter_profile(nlayers=5, epsilon=0.5, s0=1350.0, albedo=0.33)
