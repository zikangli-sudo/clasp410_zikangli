#!/usr/bin/env python3

'''
This files solves the N-layer atmosphere problem for Lab 01.

'''

import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
sigma = 5.67E-8  #Units: W/m2/K-4

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350, debug=False):
    """
    Solve the N-layer gray-atmosphere radiative equilibrium with
    shortwave absorbed at the surface.

    The longwave system is written as A x = b with unknowns
        x = [x0, x1, ..., xN], where
          x0 = σ Ts^4                 (surface blackbody flux),
          xi = ε σ Ti^4  (i ≥ 1)      (layer i graybody flux),
    uniform emissivity ε for all layers, and σ the Stefan-Boltzmann constant.

    Coefficients (teacher/lecture form):
        A[0,0] = -1                           # surface emits upward only
        A[i,i] = -2 for i ≥ 1                 # layer emits up & down
        A[i,j] = ε^(i>0) * (1-ε)^(|i-j|-1) for i ≠ j
    Right-hand side b has only the surface forced by shortwave:
        b[0] = - S,  with  S = S0 (1 - α) / 4 ; other rows are 0.

    After solving, temperatures are recovered by
        Ts = (x0/σ)^(1/4),
        Ti = (xi/(εσ))^(1/4) for i ≥ 1.

    Parameters
    ----------
    nlayers : int
        Number of atmospheric layers N (N ≥ 0).
    epsilon : float, default=1
        Longwave emissivity ε of each layer (uniform).
    albedo : float, default=0.33
        Planetary (Bond) albedo α.
    s0 : float, default=1350
        Solar constant S0 in W m^-2.
    debug : bool, default=False
        If True, print the assembled A matrix for inspection.

    Returns
    -------
    temps : np.ndarray, shape (N+1,)
        Temperatures [Ts, T1, ..., TN] in Kelvin.
    """

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



# --- Q5 helper: nuclear-winter variant (shortwave absorbed by TOP layer) ---
def n_layer_atmos_sw_at_top(nlayers, epsilon=0.5, albedo=0.33, s0=1350.0):
    """
    Nuclear-winter variant: same longwave matrix A as the base model, but
    **move the shortwave source to the top layer** so that no SW reaches
    the surface (top layer absorbs all incoming SW).

    Formulation:
        • A is identical to `n_layer_atmos` (same ε, same coupling).
        • RHS b is zero everywhere except the TOP (index N):
              b[N] = - S,  S = S0 (1-α) / 4
          (contrast with the baseline where b[0] = -S).

    Temperatures are recovered as in the baseline:
        Ts = (x0/σ)^(1/4),  Ti = (xi/(εσ))^(1/4) for i≥1.

    Parameters
    ----------
    nlayers : int
        Number of layers N (N ≥ 1 recommended for this scenario).
    epsilon : float, default=0.5
        Longwave emissivity ε of each layer (uniform).
    albedo : float, default=0.33
        Planetary albedo α.
    s0 : float, default=1350.0
        Solar constant S0 in W m^-2.

    Returns
    -------
    temps : np.ndarray, shape (N+1,)
        Temperatures [Ts, T1, ..., TN] in Kelvin.
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
    """
    Q3(a): With a single atmospheric layer (N=1), sweep emissivity ε ∈ (0,1)
    and plot the surface temperature Ts(ε). Also print the ε that gives Ts
    closest to `target_ts`.

    Parameters
    ----------
    s0 : float, default=1350.0
        Solar constant S0 in W m^-2.
    albedo : float, default=0.33
        Planetary albedo α used in the sweep.
    target_ts : float, default=288.0
        Target surface temperature (K) to mark with a horizontal line.

    Returns
    -------
    None
        Creates a Matplotlib figure (shown on screen) and prints the
        ε* that minimizes |Ts(ε) - target_ts| along with the Ts value.
    """
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
    """
    Q3(b): Fix emissivity at `epsilon_fixed` and vary the number of layers N.
    Plot Ts(N), report (i) the N whose Ts is closest to `target_ts`, and
    (ii) the minimal N such that Ts ≥ `target_ts`. Also plot a temperature–
    altitude profile for the chosen N (illustrative altitude spacing).

    Parameters
    ----------
    epsilon_fixed : float, default=0.255
        Fixed longwave emissivity ε for all layers.
    s0 : float, default=1350.0
        Solar constant S0 in W m^-2.
    albedo : float, default=0.33
        Planetary albedo α.
    target_ts : float, default=288.0
        Target surface temperature (K).

    Returns
    -------
    None
        Produces two Matplotlib figures:
          1) Ts vs N curve with the target line,
          2) Temperature–altitude profile for the selected N.
        Prints both the “closest N” and the “minimal N with Ts ≥ target”.
    """
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

    N_ge = next((int(N) for N, Ts in zip(N_vals, Ts_by_N) if Ts >= target_ts), None)
    print("[Q3(b)] closest N = %d (Ts ~ %.2f K); minimal N with Ts ≥ %.0f K: %s"
          % (N_closest, Ts_N_closest, target_ts, str(N_ge)))


# -------------- Q4: Ts vs N curve (ε=1; default is Venus) -----------
def q4_ts_vs_layers_curve(s0=2600.0, albedo=0.75, target_ts=700.0, Nmax=120, title_note="Venus"):
    """
    Q4: With ε = 1 (perfect longwave absorption in each layer), compute and
    plot Ts as a function of N for the specified (S0, α), and mark the target
    temperature (default 700 K for Venus). Report both the N closest to the
    target and the minimal N such that Ts ≥ target.

    Parameters
    ----------
    s0 : float, default=2600.0
        Solar constant S0 in W m^-2 (Venus uses ~2600).
    albedo : float, default=0.75
        Planetary albedo α (Venus ~0.75). Change to 0.33 for Earth-like runs.
    target_ts : float, default=700.0
        Target surface temperature (K) to mark.
    Nmax : int, default=120
        Maximum N to include on the horizontal axis.
    title_note : str, default="Venus"
        A short label inserted into the plot title (e.g., "Venus" or "Earth").

    Returns
    -------
    None
        Creates a Matplotlib figure and prints both:
          • the N with Ts closest to the target,
          • the minimal N with Ts ≥ target.
    """
    epsilon = 1.0
    N_vals = list(range(0, Nmax + 1))
    Ts_by_N = np.array([n_layer_atmos(N, epsilon=epsilon, albedo=albedo, s0=s0)[0] for N in N_vals])

    idx_closest = int(np.argmin(np.abs(Ts_by_N - target_ts)))
    N_closest = int(N_vals[idx_closest])
    Ts_closest = float(Ts_by_N[idx_closest])

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
def q5_nuclear_winter_profile(nlayers=5, epsilon=0.5, s0=1350.0, albedo=0.33):
    """
    Q5: Nuclear-winter temperature-altitude profile. Compute temperatures
    for the case where all shortwave is absorbed by the TOP layer (no SW
    reaches the surface), then plot the vertical profile.

    Parameters
    ----------
    nlayers : int, default=5
        Number of atmospheric layers N.
    epsilon : float, default=0.5
        Longwave emissivity ε (uniform across layers).
    s0 : float, default=1350.0
        Solar constant S0 in W m^-2.
    albedo : float, default=0.33
        Planetary albedo α for this scenario.

    Returns
    -------
    None
        Produces a Matplotlib figure of temperature vs (layer index) altitude
        and prints the surface temperature in K and °C.
    """
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


# ---------------------- Run -----------------------------
if __name__ == "__main__":
    # Q3
    q3_single_layer_emissivity_curve(s0=1350.0, albedo=0.33, target_ts=288.0)
    q3_fixed_emissivity_layers_curve(epsilon_fixed=0.255, s0=1350.0, albedo=0.33, target_ts=288.0)

    # Q4
    q4_ts_vs_layers_curve(s0=2600.0, albedo=0.75, target_ts=700.0, Nmax=120, title_note="Venus")

    # Q5
    q5_nuclear_winter_profile(nlayers=5, epsilon=0.5, s0=1350.0, albedo=0.33)
