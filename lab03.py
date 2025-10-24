#!/usr/bin/env python3

'''
Lab 3 — Integrated Script (Q1-Q3)

This single file collects all code needed for:
  • Q1 — FTCS solver for the 1-D heat equation + validation against a reference array
  • Q2 — Permafrost experiment for Kangerlussuaq, Greenland (steady periodic solution)
  • Q3 — Warming scenarios for Kangerlussuaq (+0.5, +1, +3 °C) and their impacts

How to run everything
---------------------
$ python3 lab03.py
    - Generates the full set of figures for Q1-Q3 in sequence.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Reference solution for the Q1 benchmark (table from the handout)
sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
           [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
           [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
           [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
           [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
           [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
           [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
           [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
           [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
           [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
           [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
sol10p3 = np.array(sol10p3).transpose()


def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, lowerbound=0,
               upperbound=0):
    '''
    Q1 — FTCS solver for the 1-D heat equation (Forward-Time, Centered-Space).

    PDE:
        ∂U/∂t = c^2 ∂²U/∂x²

    Scheme:
        U_i^{j+1} = (1 - 2r) U_i^j + r (U_{i-1}^j + U_{i+1}^j),
        where r = c^2 * Δt / Δx^2.

    Boundary handling:
      - lowerbound/upperbound is None  → Neumann (zero gradient), boundary equals neighbor
      - lowerbound/upperbound is scalar → Dirichlet constant
      - lowerbound/upperbound is callable(time)->float → Dirichlet time-varying

    Parameters
    ----------
    xstop : float
        Spatial domain end (meters), domain is [0, xstop].
    tstop : float
        Final time (seconds), time is [0, tstop].
    dx : float
        Spatial step (meters).
    dt : float
        Time step (seconds).
    c2 : float
        Thermal diffusivity (m^2/s).
    lowerbound, upperbound : None | float | callable
        Boundary condition specifications (see above).

    Returns
    -------
    t, x : 1D numpy arrays
        Time and space grids.
    U : 2D numpy array (shape: nSpace x nTime)
        Temperature solution U(x_i, t_j).

    Notes
    -----
    • Stability check: raises ValueError if r > 0.5.
    • Q1 uses U(x,0)=4x-4x^2 as the initial condition and compares to sol10p3.
    '''

    # FTCS stability check: require r <= 0.5
    dt_max = dx**2 / (2*c2) / dt
    if dt > dt_max:
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}.')

    # Grid sizes (include both endpoints)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Coordinate arrays
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Allocate solution and set initial condition (Q1: U(x,0)=4x−4x^2)
    U = np.zeros([M, N])
    U[:, 0] = 4*x - 4*x**2

    # Diffusion number
    r = c2 * (dt/dx**2)

    # March forward in time
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Lower boundary
        if lowerbound is None:            # Neumann (zero-gradient)
            U[0, j+1] = U[1, j+1]
        elif callable(lowerbound):        # Dirichlet (time-varying)
            U[0, j+1] = lowerbound(t[j+1])
        else:                             # Dirichlet (constant)
            U[0, j+1] = lowerbound

        # Upper boundary
        if upperbound is None:            # Neumann (zero-gradient)
            U[-1, j+1] = U[-2, j+1]
        elif callable(upperbound):        # Dirichlet (time-varying)
            U[-1, j+1] = upperbound(t[j+1])
        else:                             # Dirichlet (constant)
            U[-1, j+1] = upperbound

    return t, x, U


def plot_heatsolve(t, x, U, title=None, **kwargs):
    '''
    Q1 — Quick pcolor visualization for a heat-equation solution.

    Parameters
    ----------
    t, x : 1D numpy arrays
        Time and space grids.
    U : 2D numpy array (nSpace x nTime)
        Temperature field.
    title : str or None
        Figure title.
    **kwargs :
        Passed to matplotlib.axes.Axes.pcolor (e.g., cmap, vmin, vmax).

    Returns
    -------
    fig, ax, cbar : matplotlib objects
        Figure, axes, and attached colorbar.
    '''

    # Default colormap if none provided
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Render the color map
    contour = ax.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Labels and title
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax, cbar

# 1) Run the solver with the Q1 benchmark setup (defaults match the prompt)
t, x, U = solve_heat()

# 2) Compare to the provided solution: max-norm error and allclose() test
max_err = np.max(np.abs(U - sol10p3))
ok = np.allclose(U, sol10p3, atol=1e-3)

print(f"Q1 validation -> max |error| = {max_err:.6f}, allclose={ok}")



"""
Q2 — Heat diffusion & permafrost (Kangerlussuaq, Greenland)

Purpose
-------
• March the 1-D heat equation forward year by year (daily time step) with sinusoidal
  surface forcing matching the Kangerlussuaq climatology.
• Detect periodic steady state using a deep-zone (≥60 m) year-to-year max |ΔT| criterion.
• Produce four diagnostic figures:
    Fig 1 - Deep-zone year-to-year max ΔT vs year (steady-state criterion)
    Fig 2 - Heatmap over the last 5 years near steady state
    Fig 3 - Seasonal envelopes (winter min, summer max) in the final year
    Fig 4 - Heatmap from t=0 to the first steady year (full-history view)
"""

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# -----------------------------
# Physical constants and surface forcing (from the handout)
# -----------------------------

# Thermal diffusivity for permafrost:
#   0.25 mm^2/s = 0.25e-6 m^2/s. Convert to m^2/day to match our time unit.
C2_M2_PER_S = 0.25e-6
C2 = C2_M2_PER_S * 86400.0  # ~0.0216 m^2/day

# Monthly climatology for Kangerlussuaq (provided)
T_KANGER = np.array([
    -19.7, -21.0, -17.0, -8.4,  2.3,  8.4,
     10.7,   8.5,   3.1, -6.0, -12.0, -16.9
])

def temp_kanger(t_days: np.ndarray) -> np.ndarray:
    """
    Continuous surface temperature forcing (°C) for Kangerlussuaq.
    Matches the handout: amp * sin(pi/180 * t − pi/2) + mean.
    We treat one climatic year as 365 days for indexing the last-year envelopes.
    """
    t_amp = (T_KANGER - T_KANGER.mean()).max()
    return t_amp * np.sin(np.pi/180.0 * t_days - np.pi/2.0) + T_KANGER.mean()


def ensure_ftcs_stability(dx: float, dt_days: float, c2: float) -> float:
    """
    FTCS stability check: r = c^2 * dt / dx^2 ≤ 0.5.
    Raises ValueError if violated; returns r otherwise.
    """
    r = c2 * dt_days / (dx**2)
    if r > 0.5:
        raise ValueError(
            f"Unstable FTCS configuration: r = {r:.4f} > 0.5. "
            f"Reduce dt or increase dx."
        )
    return r

def linear_zero_cross_depth(profile: np.ndarray, depths: np.ndarray) -> float:
    """
    Depth of the first 0°C crossing via linear interpolation between the first sign change.
    Returns NaN if no crossing occurs within the domain.
    """
    s = np.sign(profile)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    x0, x1 = depths[i], depths[i+1]
    y0, y1 = profile[i], profile[i+1]
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)

# -----------------------------
# One-year march + steady-state driver
# -----------------------------

def run_one_year(U0: np.ndarray, depths: np.ndarray, day0: float,
                 dx: float, dt_days: float, c2: float,
                 bottom_const_C: float, year_days: int) -> np.ndarray:
    """
    Integrate one full year with FTCS and Dirichlet boundaries:
      • Surface (depth=0): time-varying Dirichlet = temp_kanger(current_day)
      • Bottom  (depth=H): constant Dirichlet = bottom_const_C

    Returns U_year with shape (M, steps+1), including the initial column.
    """
    steps = int(year_days / dt_days)
    M = depths.size
    U = np.zeros((M, steps + 1))
    U[:, 0] = U0

    r = c2 * dt_days / (dx**2)  # guaranteed stable by external check

    for j in range(steps):
        # Interior update
        U[1:-1, j+1] = (1 - 2*r) * U[1:-1, j] + r * (U[2:, j] + U[:-2, j])
        # Surface boundary (time-varying)
        U[0, j+1] = temp_kanger(day0 + (j+1) * dt_days)
        # Bottom boundary (constant)
        U[-1, j+1] = bottom_const_C

    return U

def reach_periodic_steady_state(cfg: SimpleNamespace) -> SimpleNamespace:
    """
    March year by year from U(x,0)=0°C until the deep-zone (≥60 m) year-to-year
    maximum absolute difference falls below a specified tolerance.

    Returns:
        x                : (M,) depths (m)
        U_last_year      : (M, S) temperatures for the last simulated year
        t_window_years   : (K,) time (years) for the last-5-year window
        U_window         : (M, K) temps for the last-5-year window (for heatmap)
        years_to_steady  : float, first year index with metric < tolerance
        y2y_metric       : list of deep-zone max |ΔT| per year
        yr_summer_0C     : annual summer 0°C depths (diagnostic)
        yr_winter_0C     : annual winter 0°C depths (diagnostic)
    """
    # Unpack configuration
    H          = cfg.x_max
    dx         = cfg.dx
    dt_days    = cfg.dt_days
    c2         = cfg.c2
    bottom_C   = cfg.bottom_C
    tol        = cfg.steady_tol_C
    z_min      = cfg.steady_min_depth_m
    max_years  = cfg.max_years
    year_days  = cfg.year_days  # 365

    # Space grid
    M = int(H / dx) + 1
    x = np.linspace(0.0, H, M)

    # Stability check
    ensure_ftcs_stability(dx, dt_days, c2)

    # Initial condition: U(x,0) = 0°C
    U_state = np.zeros(M)

    # Rolling buffer for the last five years (for Fig. 2)
    keep_steps = int(5 * year_days / dt_days) + 1
    U_window = np.zeros((M, 1)) + 0.0
    t_window_days = np.array([0.0])

    # Diagnostics
    y2y_metric = []      # deep-zone year-to-year max |ΔT|
    yr_summer_0C = []    # per-year summer 0°C depth
    yr_winter_0C = []    # per-year winter 0°C depth

    prev_year_full = None
    total_days = 0
    years_to_steady = None

    for k in range(1, max_years + 1):
        # Simulate one climatic year
        U_year = run_one_year(U_state, x, day0=total_days, dx=dx, dt_days=dt_days,
                              c2=c2, bottom_const_C=bottom_C, year_days=year_days)
        total_days += year_days
        U_state = U_year[:, -1].copy()

        # Seasonal envelopes for this year (for 0°C depth diagnostics)
        winter_prof = U_year[:, 1:].min(axis=1)
        summer_prof = U_year[:, 1:].max(axis=1)
        yr_winter_0C.append(linear_zero_cross_depth(winter_prof, x))
        yr_summer_0C.append(linear_zero_cross_depth(summer_prof, x))

        # Deep-zone year-to-year metric
        if prev_year_full is not None:
            deep = x >= z_min
            diff = np.max(np.abs(U_year[deep, 1:] - prev_year_full[deep, 1:]))
            y2y_metric.append(diff)
            if years_to_steady is None and diff < tol:
                years_to_steady = k  # first year index below tolerance

        prev_year_full = U_year.copy()

        # Maintain the "last-5-year" window for the steady-state heatmap
        steps_this_year = U_year.shape[1] - 1
        days_vec = np.arange(1, steps_this_year + 1) * dt_days
        t_window_days = np.concatenate([t_window_days,
                                        t_window_days[-1] + days_vec])
        U_window = np.concatenate([U_window, U_year[:, 1:]], axis=1)
        if t_window_days.size > keep_steps:
            t_window_days = t_window_days[-keep_steps:]
            U_window = U_window[:, -keep_steps:]

        # After detection, continue a few extra years to make the window solidly steady
        if (years_to_steady is not None) and (k >= years_to_steady + 5):
            break

    if years_to_steady is None:
        years_to_steady = k  # fallback

    return SimpleNamespace(
        x=x,
        U_last_year=prev_year_full,
        t_window_years=t_window_days / year_days,
        U_window=U_window,
        years_to_steady=float(years_to_steady),
        y2y_metric=y2y_metric,
        yr_summer_0C=yr_summer_0C,
        yr_winter_0C=yr_winter_0C
    )

# -----------------------------
# Main: run, plot, and print
# -----------------------------
if __name__ == "__main__":
    # Configuration (365-day year indexing as in the handout)
    cfg = SimpleNamespace(
        x_max=100.0,          # depth domain: 0–100 m
        dx=0.5,               # spatial step (m)
        dt_days=1.0,          # time step (days)
        c2=C2,                # m^2/day
        bottom_C=5.0,         # geothermal BC at 100 m
        steady_tol_C=0.01,    # deep-zone tolerance (°C)
        steady_min_depth_m=60.0,
        max_years=500,        # safety cap
        year_days=365         # length of a climatic year
    )

    # Run to periodic steady state and collect outputs
    OUT = reach_periodic_steady_state(cfg)

    # ----------- Fig 1: Deep-zone year-to-year max difference -----------
    plt.ioff()  # keep figures separate
    plt.figure(num="Fig 1 – Deep-zone year-to-year max ΔT", figsize=(8.5, 5.3))
    years_axis = np.arange(2, 2 + len(OUT.y2y_metric))  # first diff is year 2 vs 1
    if len(years_axis) > 0:
        plt.plot(years_axis, OUT.y2y_metric, marker='o', linewidth=1.6, label="Max |ΔT| (depth ≥ 60 m)")
    plt.axhline(cfg.steady_tol_C, linestyle='--', linewidth=1.2, color='tab:red',
                label=f"Tolerance = {cfg.steady_tol_C} °C")
    plt.axvline(OUT.years_to_steady, linestyle=':', linewidth=1.2, color='k',
                label=f"First year below tol = {OUT.years_to_steady:.0f}")
    plt.title("Deep-Zone Year-to-Year Max Difference (Steady-State Criterion)")
    plt.xlabel("Year")
    plt.ylabel("Max |ΔT| over depth ≥ 60 m (°C)")
    plt.legend()
    plt.tight_layout()

    # ----------- Fig 2: Heatmap (steady-state window: last 5 years) -----------
    plt.figure(num="Fig 2 – Heatmap (steady-state window)", figsize=(9.6, 6.6))
    pc = plt.pcolor(OUT.t_window_years, OUT.x, OUT.U_window,
                    cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(pc, label="Temperature (°C)")
    plt.title("Ground Temperature: Kangerlussuaq, Greenland (Steady-State Window)")
    plt.xlabel("Time (Years)")
    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # ----------- Fig 3: Seasonal profiles (last year at steady state) -----------
    # Seasonal envelopes (winter=min, summer=max) from the last simulated year
    winter_last = OUT.U_last_year[:, 1:].min(axis=1)
    summer_last = OUT.U_last_year[:, 1:].max(axis=1)

    # 0°C crossing depths for the last year (active layer top & permafrost bottom)
    active_layer_depth = linear_zero_cross_depth(summer_last, OUT.x)
    permafrost_bottom  = linear_zero_cross_depth(winter_last, OUT.x)

    plt.figure(num="Fig 3 – Seasonal profiles (last year)", figsize=(8.6, 6.3))
    plt.plot(winter_last, OUT.x, label="Winter (min over last year)")
    plt.plot(summer_last, OUT.x, linestyle="--", label="Summer (max over last year)")
    plt.axvline(0.0, color='k', linewidth=1.0, alpha=0.6)
    # Annotate depth markers if defined
    if np.isfinite(active_layer_depth):
        plt.hlines(active_layer_depth,
                   xmin=min(winter_last.min(), summer_last.min()),
                   xmax=0.0, colors='gray', linestyles=':', linewidth=1.0)
        plt.text(0.02, active_layer_depth,
                 f"Active layer depth ≈ {active_layer_depth:.2f} m", va='center')
    if np.isfinite(permafrost_bottom):
        plt.hlines(permafrost_bottom,
                   xmin=min(winter_last.min(), summer_last.min()),
                   xmax=0.0, colors='gray', linestyles=':', linewidth=1.0)
        plt.text(0.02, permafrost_bottom,
                 f"Permafrost bottom ≈ {permafrost_bottom:.2f} m", va='center')
    plt.title("Seasonal Temperature Profiles (Last Year at Steady State)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    plt.legend(loc="best")
    plt.tight_layout()

    # ----------- Fig 4: Heatmap from t=0 to first steady year -----------
    # Reconstruct full history up to the detected steady year
    steady_year = int(np.ceil(OUT.years_to_steady))

    M = int(cfg.x_max / cfg.dx) + 1
    x_full = np.linspace(0.0, cfg.x_max, M)
    U_state = np.zeros(M)

    dt = cfg.dt_days
    steps_per_year = int(cfg.year_days / dt)
    total_steps = steady_year * steps_per_year

    U_hist = np.zeros((M, total_steps + 1))
    U_hist[:, 0] = U_state
    t_hist_days = np.zeros(total_steps + 1)

    day0 = 0.0
    col = 0
    for y in range(steady_year):
        U_year = run_one_year(
            U0=U_state, depths=x_full, day0=day0,
            dx=cfg.dx, dt_days=cfg.dt_days, c2=cfg.c2,
            bottom_const_C=cfg.bottom_C, year_days=cfg.year_days
        )
        steps = U_year.shape[1] - 1
        U_hist[:, col + 1: col + 1 + steps] = U_year[:, 1:]
        t_hist_days[col + 1: col + 1 + steps] = day0 + np.arange(1, steps + 1) * dt

        day0 += cfg.year_days
        col += steps
        U_state = U_year[:, -1].copy()

    t_hist_years = t_hist_days / cfg.year_days

    plt.figure(num="Fig 4 – Heatmap (0 to steady year)", figsize=(10.0, 6.5))
    pc = plt.pcolor(t_hist_years, x_full, U_hist, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(pc, label="Temperature (°C)")
    plt.title("Ground Temperature: Kangerlussuaq, Greenland (0 to first steady year)")
    plt.xlabel("Time (Years)")
    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # ----------- Print Q2 answers and figure mapping -----------
    print("\n=== Q2 Outputs (steady state) ===")
    print(f"Years to reach steady state (first year below tolerance): {OUT.years_to_steady:.0f} years")
    print(f"Active layer depth (summer 0°C crossing, last year):       {active_layer_depth:.2f} m")
    print(f"Permafrost top (same as active layer depth):               {active_layer_depth:.2f} m")
    print(f"Permafrost bottom (winter 0°C crossing, last year):        {permafrost_bottom:.2f} m")

    print("\n[Figure mapping]")
    print("Fig 1: Steady-state criterion — deep-zone year-to-year max difference vs. year.")
    print("Fig 2: Heatmap (steady-state window) — last 5 years after steady state.")
    print("Fig 3: Seasonal profiles (last year) — used to read active layer and permafrost depths.")
    print("Fig 4: 0°C depth trajectories — optional convergence view over years.")

    # Show all figures
    plt.show()



"""
Q3 — Warming scenarios (+0.5, +1, +3 °C) for Kangerlussuaq

Purpose
-------
• Repeat the Q2 permafrost setup but add a uniform surface warming shift ΔT.
• For each ΔT ∈ {+0.5, +1.0, +3.0} °C:
    - March to periodic steady state
    - Extract final-year seasonal envelopes
    - Compute active layer depth (summer 0°C), permafrost bottom (winter 0°C),
      and permafrost thickness (bottom - active)
• Produce:
    Fig A/B/C - Seasonal profiles for each ΔT at the final steady year (with annotations)
    Fig D     - Trends: active layer depth vs ΔT, permafrost thickness vs ΔT

All plots are displayed with plt.show().
"""

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# -----------------------------
# Physical constants & forcing (as in Q2)
# -----------------------------

# Permafrost thermal diffusivity: 0.25 mm^2/s = 0.25e-6 m^2/s
# Convert to m^2/day to match the day-based time step.
C2_M2_PER_S = 0.25e-6
C2 = C2_M2_PER_S * 86400.0  # ≈ 0.0216 m^2/day

# Monthly climatology for Kangerlussuaq (provided)
T_KANGER = np.array([
    -19.7, -21.0, -17.0, -8.4,  2.3,  8.4,
     10.7,   8.5,   3.1, -6.0, -12.0, -16.9
])

def temp_kanger(t_days: np.ndarray) -> np.ndarray:
    """
    Continuous surface temperature forcing (°C) for Kangerlussuaq,
    as in the handout: amp * sin(pi/180 * t − pi/2) + mean.
    """
    t_amp = (T_KANGER - T_KANGER.mean()).max()
    return t_amp * np.sin(np.pi/180.0 * t_days - np.pi/2.0) + T_KANGER.mean()

def temp_kanger_shifted(t_days: np.ndarray, dT: float) -> np.ndarray:
    """Apply a uniform warming shift ΔT (°C) to the surface temperature."""
    return temp_kanger(t_days) + dT


def ensure_ftcs_stability(dx: float, dt_days: float, c2: float) -> float:
    """FTCS stability: require r = c^2 * dt / dx^2 ≤ 0.5. Raise if violated."""
    r = c2 * dt_days / (dx**2)
    if r > 0.5:
        raise ValueError(
            f"Unstable FTCS configuration: r = {r:.4f} > 0.5. "
            "Reduce dt or increase dx."
        )
    return r

def linear_zero_cross_depth(profile: np.ndarray, depths: np.ndarray) -> float:
    """
    Linear interpolation for the first 0 °C crossing depth.
    Returns NaN if no sign change occurs within the domain.
    """
    s = np.sign(profile)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    x0, x1 = depths[i], depths[i+1]
    y0, y1 = profile[i], profile[i+1]
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)

# -----------------------------
# Core solver (year march + steady-state detection)
# -----------------------------

def run_one_year(U0: np.ndarray, depths: np.ndarray, day0: float,
                 dx: float, dt_days: float, c2: float,
                 bottom_const_C: float, year_days: int,
                 surface_fun) -> np.ndarray:
    """
    Integrate a single year with FTCS:
      - Surface (0 m): Dirichlet = surface_fun(time)
      - Bottom  (H):  Dirichlet = bottom_const_C
    Returns U_year with shape (M, steps+1), including the initial column.
    """
    steps = int(year_days / dt_days)
    M = depths.size
    U = np.zeros((M, steps + 1))
    U[:, 0] = U0
    r = c2 * dt_days / (dx**2)  # stability assumed checked externally

    for j in range(steps):
        # Interior update
        U[1:-1, j+1] = (1 - 2*r) * U[1:-1, j] + r * (U[2:, j] + U[:-2, j])
        # Boundary updates
        U[0,  j+1] = surface_fun(day0 + (j+1) * dt_days)
        U[-1, j+1] = bottom_const_C

    return U

def reach_periodic_steady_state(cfg: SimpleNamespace, dT: float) -> SimpleNamespace:
    """
    Starting from U(x,0)=0 °C, march year by year with a uniform surface shift ΔT,
    until the deep-zone (≥60 m) year-to-year max |ΔT| is below the tolerance.

    Returns:
        x               : (M,) depths
        U_last_year     : (M, S) temperatures for the last simulated year
        years_to_steady : first year index below tolerance
        active_depth    : summer 0 °C crossing (permafrost top)
        permafrost_bot  : winter 0 °C crossing (permafrost bottom)
        thickness       : permafrost thickness (bottom - active)
        winter_last     : winter envelope of last year
        summer_last     : summer envelope of last year
    """
    H, dx, dt, c2 = cfg.x_max, cfg.dx, cfg.dt_days, cfg.c2
    bottom_C = cfg.bottom_C
    tol = cfg.steady_tol_C
    z_min = cfg.steady_min_depth_m
    max_years = cfg.max_years
    year_days = cfg.year_days

    # Grid & stability
    M = int(H / dx) + 1
    x = np.linspace(0.0, H, M)
    ensure_ftcs_stability(dx, dt, c2)

    # Initial condition
    U_state = np.zeros(M)
    prev_year_full = None
    total_days = 0
    years_to_steady = None

    # Surface forcing with ΔT shift
    surf = lambda t: temp_kanger_shifted(t, dT)

    for k in range(1, max_years + 1):
        U_year = run_one_year(U_state, x, day0=total_days, dx=dx, dt_days=dt,
                              c2=c2, bottom_const_C=bottom_C, year_days=year_days,
                              surface_fun=surf)
        total_days += year_days
        U_state = U_year[:, -1].copy()

        if prev_year_full is not None:
            deep = x >= z_min
            diff = np.max(np.abs(U_year[deep, 1:] - prev_year_full[deep, 1:]))
            if years_to_steady is None and diff < tol:
                years_to_steady = k
        prev_year_full = U_year.copy()

        # Run a few extra years after detection to ensure robust final-year envelopes
        if (years_to_steady is not None) and (k >= years_to_steady + 5):
            break

    if years_to_steady is None:
        years_to_steady = k

    # Final-year seasonal envelopes
    winter_last = prev_year_full[:, 1:].min(axis=1)
    summer_last = prev_year_full[:, 1:].max(axis=1)

    # Metrics
    active = linear_zero_cross_depth(summer_last, x)
    bottom = linear_zero_cross_depth(winter_last, x)
    thickness = bottom - active if np.isfinite(active) and np.isfinite(bottom) else np.nan

    return SimpleNamespace(
        x=x,
        U_last_year=prev_year_full,
        years_to_steady=float(years_to_steady),
        active_depth=float(active),
        permafrost_bot=float(bottom),
        thickness=float(thickness),
        winter_last=winter_last,
        summer_last=summer_last
    )

# -----------------------------
# Plot helpers
# -----------------------------

def plot_seasonal_profile(x, winter, summer, title_suffix, xlim=(-25, 10), ylim=(0, 100)):
    """
    One seasonal profile panel with annotations:
      - Active layer depth (summer 0 °C crossing)
      - Permafrost bottom (winter 0 °C crossing)
      - Thickness (difference)
    Keeps axes consistent across scenarios for fair visual comparison.
    """
    active = linear_zero_cross_depth(summer, x)
    bottom = linear_zero_cross_depth(winter, x)
    thickness = bottom - active if np.isfinite(active) and np.isfinite(bottom) else np.nan

    plt.figure(figsize=(7.8, 6.2))
    plt.plot(winter, x, label="Winter (min over last year)")
    plt.plot(summer, x, '--', label="Summer (max over last year)")
    plt.axvline(0.0, color='k', lw=1.0, alpha=0.7)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Depth (m)")
    plt.title(f"Seasonal Profiles (Last Steady Year) — {title_suffix}")
    plt.legend(loc="lower left")

    # Annotations
    if np.isfinite(active):
        plt.hlines(active, xlim[0], 0, colors='gray', linestyles=':', lw=1.0)
        plt.text(xlim[0] + 1, active, f"Active layer ≈ {active:.2f} m", va='center')
    if np.isfinite(bottom):
        plt.hlines(bottom, xlim[0], 0, colors='gray', linestyles=':', lw=1.0)
        plt.text(xlim[0] + 1, bottom, f"Permafrost bottom ≈ {bottom:.2f} m", va='center')
    if np.isfinite(active) and np.isfinite(bottom):
        plt.text(xlim[0] + 1, (active + bottom)/2,
                 f"Thickness ≈ {thickness:.2f} m", va='center', fontsize=10)

    plt.tight_layout()
    return active, bottom, thickness

# -----------------------------
# Main (Q3)
# -----------------------------
if __name__ == "__main__":
    # Shared configuration (same as Q2; only the surface is shifted by ΔT)
    cfg = SimpleNamespace(
        x_max=100.0,         # depth domain (m)
        dx=0.5,              # spatial step (m)
        dt_days=1.0,         # time step (day)
        c2=C2,               # m^2/day
        bottom_C=5.0,        # geothermal boundary at 100 m
        steady_tol_C=0.01,   # deep-zone tolerance (°C)
        steady_min_depth_m=60.0,
        max_years=500,
        year_days=365
    )

    # Warming scenarios
    deltas = [0.5, 1.0, 3.0]

    # Arrays for trend figure (Fig D)
    actives, bottoms, thicknesses = [], [], []

    # Fixed axes for comparable seasonal-profile figures
    X_LIM = (-25, 10)
    Y_LIM = (0, 100)

    results = {}

    for dT in deltas:
        OUT = reach_periodic_steady_state(cfg, dT=dT)
        results[dT] = OUT

        # Fig A/B/C: seasonal profiles at the last steady year
        title_suffix = f"ΔT = +{dT:.1f} °C"
        a, b, th = plot_seasonal_profile(OUT.x, OUT.winter_last, OUT.summer_last,
                                         title_suffix=title_suffix, xlim=X_LIM, ylim=Y_LIM)
        actives.append(a)
        bottoms.append(b)
        thicknesses.append(th)

        print(f"[ΔT=+{dT:.1f} °C] steady by ~year {OUT.years_to_steady:.0f} | "
              f"Active={a:.2f} m, Bottom={b:.2f} m, Thickness={th:.2f} m")

    # Fig D: trends with ΔT
    plt.figure(figsize=(11.0, 5.2))
    # Left: Active layer depth vs ΔT
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(deltas, actives, marker='o', lw=1.8)
    ax1.set_xlabel("ΔT (°C)")
    ax1.set_ylabel("Active layer depth (m)")
    ax1.set_title("Active layer depth vs ΔT")
    ax1.grid(True, alpha=0.3)

    # Right: Permafrost thickness vs ΔT
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(deltas, thicknesses, marker='o', lw=1.8, color='tab:red')
    ax2.set_xlabel("ΔT (°C)")
    ax2.set_ylabel("Permafrost thickness (m)")
    ax2.set_title("Permafrost thickness vs ΔT")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
