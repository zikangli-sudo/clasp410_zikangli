#!/usr/bin/env python3
"""
Forest fire & disease-spread lab — Q1, Q2, Q3
- States:
  * Forest fire: 2=Forest, 3=FIRE, 1=Bare
  * Disease:     2=Healthy, 3=Sick, 1=Immune, 0=Dead

This script implements the full numerical pipeline used in the lab:
Q1: small-grid forest-fire demos to verify the rules and show time evolution.
Q2: parameter sweeps for wildfire spread (P_spread and initial forest density).
Q3: a disease-spread analog using the same grid structure.

How to run everything
---------------------
$ python3 lab03.py
    - Generates the full set of figures for Q1-Q3 in sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# -----------------------------
# Helpers & constants
# -----------------------------
def pct(series, total):
    """Return percentages for a count series given the total size.

    Parameters
    ----------
    series : array-like
        Counts at each time step or category.
    total : int or float
        Total number of cells / individuals.

    Returns
    -------
    np.ndarray
        Array of percentages (0-100) with the same length as series.
    """
    return 100.0 * np.array(series, dtype=float) / float(total)

# Color definitions for plotting the different states
COLOR_FIRE = (200/255, 30/255, 70/255)      # red
COLOR_FORE = (0/255,   100/255, 0/255)      # dark green
COLOR_BARE = (210/255, 180/255, 140/255)    # tan
COLOR_SICK = COLOR_FIRE
COLOR_HEAL = COLOR_FORE
COLOR_IMMU = (70/255,  130/255, 180/255)    # steel-blue
COLOR_DEAD = (80/255,  80/255,  80/255)     # gray

# Global random number generator for reproducibility
# All stochastic behavior in this script comes from this rng.
rng = default_rng(42)  # global seed for reproducibility

# ============================================================
# Q1 — Forest fire model
# ============================================================
def forest_fire(isize=3, jsize=3, nstep=4,
                pspread=1.0, pignite=0.0, pbare=0.0,
                *, use_center_ignite=True):
    """Forest fire cellular automaton: 2=Forest, 3=FIRE, 1=Bare. 4-neighbors.

    At each time step, burning cells (state 3) may ignite neighboring forest
    cells (state 2) with probability pspread in the four cardinal directions.
    Burning cells always turn into bare ground (state 1) at the next step.

    Parameters
    ----------
    isize, jsize : int
        Grid dimensions (rows, columns).
    nstep : int
        Number of time steps to simulate.
    pspread : float
        Probability that a burning cell ignites each forest neighbor.
    pignite : float
        Probability that a cell is initially ignited (t=0) if
        use_center_ignite is False or pignite > 0.
    pbare : float
        Probability that a cell is initially bare (no fuel).
    use_center_ignite : bool, keyword-only
        If True and pignite == 0, ignite the center cell deterministically
        at t=0. Otherwise, use probabilistic ignition based on pignite.

    Returns
    -------
    np.ndarray
        Array of shape (nstep, isize, jsize) with integer states.
    """
    # Allocate the full 3D array: time × rows × cols, initialized to Forest (2)
    grid = np.zeros((nstep, isize, jsize), dtype=int) + 2  # all Forest

    # ignition at t=0
    if use_center_ignite and pignite == 0:
        # Ignite the central cell if no probabilistic ignition is requested
        grid[0, isize//2, jsize//2] = 3
    else:
        # Random initial ignitions based on pignite
        ign = rng.random((isize, jsize)) <= pignite
        # Ensure at least one ignition by forcing the center if needed
        if ign.sum() == 0:
            ign[isize//2, jsize//2] = True
        grid[0, ign] = 3

    # initial bare patches (fuel missing)
    # Note that ignited cells may be overwritten as bare if both events happen.
    grid[0, rng.random((isize, jsize)) <= pbare] = 1

    # evolve in time
    for k in range(nstep - 1):
        # Start from previous state and update into the next time slice
        # We copy the whole slice and then modify cells that change.
        grid[k+1] = grid[k]
        for i in range(isize):
            for j in range(jsize):
                # Only burning cells can spread fire and then become bare
                if grid[k, i, j] != 3:
                    continue
                # Try to ignite each of the four neighbors (4-neighbor connectivity)
                # Spread to neighbor above
                if i > 0 and grid[k, i-1, j] == 2 and rng.random() < pspread:
                    grid[k+1, i-1, j] = 3
                # Spread to neighbor below
                if i < isize-1 and grid[k, i+1, j] == 2 and rng.random() < pspread:
                    grid[k+1, i+1, j] = 3
                # Spread to neighbor on the left
                if j > 0 and grid[k, i, j-1] == 2 and rng.random() < pspread:
                    grid[k+1, i, j-1] = 3
                # Spread to neighbor on the right
                if j < jsize-1 and grid[k, i, j+1] == 2 and rng.random() < pspread:
                    grid[k+1, i, j+1] = 3
                # burning -> bare next step (fuel consumed)
                grid[k+1, i, j] = 1
    return grid

def draw_grid_with_labels(ax, grid, title, *, title_fs=14, label_fs=10, idx_fs=8):
    """Render a small grid with color and human-readable labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the grid.
    grid : np.ndarray
        2D array for a single time slice of the forest model.
    title : str
        Title for this subplot.
    title_fs, label_fs, idx_fs : int
        Font sizes for title, state labels, and index labels.
    """
    H, W = grid.shape
    rgb = np.zeros((H, W, 3), float)
    color = {2: COLOR_FORE, 3: COLOR_FIRE, 1: COLOR_BARE}
    for i in range(H):
        for j in range(W):
            rgb[i, j] = color[int(grid[i, j])]

    ax.imshow(rgb, origin='lower', extent=[0, W, 0, H], interpolation='none')
    ax.set_title(title, fontsize=title_fs, pad=6)
    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))

    # Add text showing the state and (i, j) index inside each cell
    for i in range(H):
        for j in range(W):
            v = int(grid[i, j])
            label = 'Forest' if v == 2 else ('FIRE!' if v == 3 else 'Bare')
            # State label (Forest / FIRE! / Bare)
            ax.text(j + 0.5, i + 0.62, label, ha='center', va='center',
                    fontsize=label_fs, color='black')
            # Index label (i, j) to help connect with writeup
            ax.text(j + 0.5, i + 0.30, f'i, j = {i}, {j}',
                    ha='center', va='center', fontsize=idx_fs, color='black')

def q1_snapshots_3x3():
    """Q1 helper: run 3x3 forest-fire demo and show four snapshots.

    This reproduces the instructor's example on a 3x3 grid.
    """
    ff = forest_fire(isize=3, jsize=3, nstep=4,
                       pspread=1.0, pignite=0.0, pbare=0.0)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)
    for ax, s in zip(axes.ravel(), [0, 1, 2, 3]):
        draw_grid_with_labels(ax, ff[s],
                              f"Forest Status (iStep={s})",
                              title_fs=18, label_fs=11, idx_fs=9)
    plt.show()

def q1_wide_grid_demo():
    """Q1 helper: run 3x9 demo, show snapshots and time-series curves.

    The snapshots show how the fire front propagates.
    The time series show global percentages vs time.
    """
    isz, jsz, nstep = 3, 9, 6
    ff = forest_fire(isize=isz, jsize=jsz, nstep=nstep,
                       pspread=1.0, pignite=0.0, pbare=0.0)

    # snapshots at selected time steps
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.8), constrained_layout=True)
    for ax, s in zip(axes.ravel(), [0, 1, 2, 3]):
        draw_grid_with_labels(ax, ff[s],
                              title=f"Forest Status (iStep={s})",
                              title_fs=18, label_fs=9, idx_fs=8)
    plt.show()

    # progression curves (global state fractions vs time)
    ksize, H, W = ff.shape
    npoints = H * W
    # For each time step, compute fraction of cells in each state
    perc_forest = [100.0 * (ff[k] == 2).sum() / npoints for k in range(ksize)]
    perc_bare   = [100.0 * (ff[k] == 1).sum() / npoints for k in range(ksize)]
    perc_fire   = [100.0 * (ff[k] == 3).sum() / npoints for k in range(ksize)]

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot(range(ksize), perc_forest, label="Forested")
    ax.plot(range(ksize), perc_bare,   label="Bare/Burnt")
    ax.plot(range(ksize), perc_fire,   label="Burning")
    ax.set_title("Q1: 3x9 progression")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Percent of grid")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.show()

# ============================================================
# Q2 — Fire spread analysis
# ============================================================
def fire_stats_duration_final_peak(ff):
    """Return (duration steps with fire present, final % bare, peak % burning).

    Parameters
    ----------
    ff : np.ndarray
        Forest-fire state array of shape (nstep, isize, jsize).

    Returns
    -------
    tuple
        (duration, final_bare_pct, peak_burn_pct)
    """
    ksize, isize, jsize = ff.shape
    n = isize * jsize
    # Count number of burning and bare cells at each time step
    burning = [(ff[k] == 3).sum() for k in range(ksize)]
    bare    = [(ff[k] == 1).sum() for k in range(ksize)]
    # Duration counts how many steps have at least one burning cell
    duration = sum(b > 0 for b in burning)
    final_bare_pct = 100.0 * bare[-1] / n
    peak_burn_pct  = 100.0 * (max(burning) / n)
    return duration, final_bare_pct, peak_burn_pct

def fire_time_series_curves(ff):
    """Return (%Forest, %Bare, %Burning) time series from an ff run.

    This is used for plotting time-series evolution and for ensemble averages.
    """
    ksize, isize, jsize = ff.shape
    n = isize * jsize
    perc_forest = [100.0 * (ff[k] == 2).sum() / n for k in range(ksize)]
    perc_bare   = [100.0 * (ff[k] == 1).sum() / n for k in range(ksize)]
    perc_fire   = [100.0 * (ff[k] == 3).sum() / n for k in range(ksize)]
    return perc_forest, perc_bare, perc_fire

# ---------- time-series subplot helpers ----------
def fire_time_series_subplots_pspread(ps_list, isize, jsize, nstep,
                                        pignite, pbare, reps):
    """
    For each P_spread in ps_list, run the model several times,
    average the time series, and plot Forest/Bare/Burning
    in separate subplots.

    This shows how time evolution changes between low, medium,
    and high spread probabilities.
    """
    ncols = len(ps_list)
    fig, axes = plt.subplots(
        1, ncols, figsize=(5 * ncols, 4.5),
        sharey=True, constrained_layout=True
    )
    if ncols == 1:
        axes = [axes]

    for ax, ps in zip(axes, ps_list):
        # These arrays accumulate ensemble sums over reps realizations
        sum_f = sum_b = sum_fire = None

        for _ in range(reps):
            # Run one realization of the forest-fire model
            ff = forest_fire(
                isize=isize, jsize=jsize, nstep=nstep,
                pspread=ps, pignite=pignite, pbare=pbare,
                use_center_ignite=False
            )
            # Convert grid to time series of percentages
            fore, bare, fire = fire_time_series_curves(ff)
            f_arr  = np.array(fore)
            b_arr  = np.array(bare)
            fi_arr = np.array(fire)

            if sum_f is None:
                # First run initializes the accumulators
                sum_f, sum_b, sum_fire = f_arr, b_arr, fi_arr
            else:
                # Later runs add to the accumulators
                sum_f    += f_arr
                sum_b    += b_arr
                sum_fire += fi_arr

        # Convert sums to ensemble means
        mean_f    = sum_f / reps
        mean_b    = sum_b / reps
        mean_fire = sum_fire / reps

        # Plot the three time series for this value of P_spread
        ax.plot(mean_f,    label="Forested",   color=COLOR_FORE)
        ax.plot(mean_b,    label="Bare/Burnt", color=COLOR_BARE)
        ax.plot(mean_fire, label="Burning",    color=COLOR_FIRE)

        ax.set_title(f"P_spread = {ps:.1f}")
        ax.set_xlabel("Time step")
        ax.grid(alpha=0.25)

    # Share y-axis label and put legend above the first subplot (centered)
    axes[0].set_ylabel("% of grid")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3
    )
    plt.show()


def fire_time_series_subplots_density(density_list, isize, jsize, nstep,
                                        pspread, pignite, reps):
    """
    For each initial forest density in density_list, run the model several times,
    average the time series, and plot Forest/Bare/Burning
    in separate subplots.
    density = 1 - P_bare.

    This shows how the connectivity of fuel affects fire evolution.
    """
    ncols = len(density_list)
    fig, axes = plt.subplots(
        1, ncols, figsize=(5 * ncols, 4.5),
        sharey=True, constrained_layout=True
    )
    if ncols == 1:
        axes = [axes]

    for ax, dens in zip(axes, density_list):
        pbare = 1.0 - dens  # convert density -> P_bare
        # Ensemble accumulators for each density
        sum_f = sum_b = sum_fire = None

        for _ in range(reps):
            # Run one realization with this initial forest density
            ff = forest_fire(
                isize=isize, jsize=jsize, nstep=nstep,
                pspread=pspread, pignite=pignite, pbare=pbare,
                use_center_ignite=False
            )
            fore, bare, fire = fire_time_series_curves(ff)
            f_arr  = np.array(fore)
            b_arr  = np.array(bare)
            fi_arr = np.array(fire)

            if sum_f is None:
                sum_f, sum_b, sum_fire = f_arr, b_arr, fi_arr
            else:
                sum_f    += f_arr
                sum_b    += b_arr
                sum_fire += fi_arr

        # Average over all realizations
        mean_f    = sum_f / reps
        mean_b    = sum_b / reps
        mean_fire = sum_fire / reps

        # Plot the mean curves for this initial density
        ax.plot(mean_f,    label="Forested",   color=COLOR_FORE)
        ax.plot(mean_b,    label="Bare/Burnt", color=COLOR_BARE)
        ax.plot(mean_fire, label="Burning",    color=COLOR_FIRE)

        ax.set_title(f"Initial density = {dens:.1f}")
        ax.set_xlabel("Time step")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("% of grid")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3
    )
    plt.show()
# --------------------------------------------------------


def q2_sweep_pspread(isize=30, jsize=30, nstep=35,
                       pignite=0.02, pbare=0.0, reps=8):
    """
    Sweep P_spread in [0..1] and plot (vertical stack):
      - Duration vs P_spread
      - Final % bare vs P_spread
      - Peak % burning vs P_spread
    Then: multi-subplot time series for a few representative P_spread values.

    This addresses the Q2 part about how fire behavior changes with P_spread.
    """
    # Uniform grid of spread probabilities from 0.0 to 1.0 (inclusive)
    grid_ps = np.linspace(0.0, 1.0, 11)
    durations, finals, peaks = [], [], []

    for ps in grid_ps:
        # Run multiple realizations for each P_spread and average diagnostics
        d_list, f_list, p_list = [], [], []
        for _ in range(reps):
            ff = forest_fire(isize=isize, jsize=jsize, nstep=nstep,
                             pspread=ps, pignite=pignite, pbare=pbare,
                             use_center_ignite=False)
            di, fi, pi = fire_stats_duration_final_peak(ff)
            d_list.append(di); f_list.append(fi); p_list.append(pi)
        # Store ensemble means for this value of P_spread
        durations.append(np.mean(d_list))
        finals.append(np.mean(f_list))
        peaks.append(np.mean(p_list))

    # Summary panel: duration, final bare %, peak burning % vs P_spread
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 14),
                                       constrained_layout=True)
    ax1.plot(grid_ps, durations, marker='o')
    ax1.set_title("Duration vs P_spread", pad=8)
    ax1.set_ylabel("Duration (iterations)")
    ax1.grid(alpha=0.25); ax1.label_outer()

    ax2.plot(grid_ps, finals, marker='o')
    ax2.set_title("Final burn vs P_spread", pad=8)
    ax2.set_ylabel("Final % bare")
    ax2.grid(alpha=0.25); ax2.label_outer()

    ax3.plot(grid_ps, peaks, marker='o')
    ax3.set_title("Peak vs P_spread", pad=8)
    ax3.set_xlabel("P_spread")
    ax3.set_ylabel("Peak % burning")
    ax3.grid(alpha=0.25)
    plt.show()

    # Time-series subplots for three representative P_spread values
    ps_list = [0.2, 0.5, 0.8]
    fire_time_series_subplots_pspread(
        ps_list=ps_list,
        isize=isize, jsize=jsize, nstep=nstep,
        pignite=pignite, pbare=pbare, reps=reps
    )


def q2_sweep_pbare(isize=30, jsize=30, nstep=35,
                     pspread=0.6, pignite=0.02, reps=8):
    """
    Sweep P_bare in [0..1] (i.e., initial non-forest fraction).
    Vertical stack:
      - Duration vs initial forest density (1 - P_bare)
      - Final % burned vs initial forest density
      - Peak % burning vs initial forest density
    Then: multi-subplot time series for a few representative initial densities.

    This addresses the Q2 part about how initial fuel availability
    controls fire behavior.
    """
    # P_bare grid from 0 (fully forested) to 1 (no forest)
    grid_pb = np.linspace(0.0, 1.0, 11)
    durations, finals, peaks = [], [], []

    for pb in grid_pb:
        # For each initial bare fraction, run several realizations
        d_list, f_list, p_list = [], [], []
        for _ in range(reps):
            ff = forest_fire(isize=isize, jsize=jsize, nstep=nstep,
                             pspread=pspread, pignite=pignite, pbare=pb,
                             use_center_ignite=False)
            di, fi, pi = fire_stats_duration_final_peak(ff)
            d_list.append(di); f_list.append(fi); p_list.append(pi)
        durations.append(np.mean(d_list))
        finals.append(np.mean(f_list))
        peaks.append(np.mean(p_list))

    # Convert P_bare to initial forest density (fraction of fuel)
    dens = 1.0 - grid_pb  # initial forest density
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 14),
                                       constrained_layout=True)
    ax1.plot(dens, durations, marker='o'); ax1.set_title(
        "Duration vs initial forest density", pad=8)
    ax1.set_ylabel("Duration (iterations)")
    ax1.grid(alpha=0.25); ax1.label_outer()

    ax2.plot(dens, finals, marker='o'); ax2.set_title(
        "Final burn vs initial forest density", pad=8)
    ax2.set_ylabel("Final % burned")
    ax2.grid(alpha=0.25); ax2.label_outer()

    ax3.plot(dens, peaks, marker='o'); ax3.set_title(
        "Peak vs initial forest density", pad=8)
    ax3.set_xlabel("Initial forest density")
    ax3.set_ylabel("Peak % burning")
    ax3.grid(alpha=0.25)
    plt.show()

    # Time-series subplots for three representative densities
    density_list = [0.2, 0.5, 0.8]    # 20%, 50%, 80% forest
    fire_time_series_subplots_density(
        density_list=density_list,
        isize=isize, jsize=jsize, nstep=nstep,
        pspread=pspread, pignite=pignite, reps=reps
    )

# ============================================================
# Q3 — Disease-spread analog
# ============================================================
def disease_spread(isize=40, jsize=40, nstep=60,
                   pspread=0.6, pignite=0.02,
                   p_survive=0.8, p_vax=0.2):
    """
    Disease model (4 states): 0=Dead, 1=Immune, 2=Healthy, 3=Sick.

    Each step:
      - Sick infect 4-neighbor Healthy with prob pspread.
      - Then Sick resolve: Immune with prob p_survive else Dead.
    t=0: fraction p_vax are Immune; fraction pignite are Sick.

    This reuses the same grid structure as the fire model but with
    health states instead of fuel states.
    """
    # Allocate full state history: time × rows × cols, initialized as Healthy
    grid = np.zeros((nstep, isize, jsize), dtype=int) + 2  # all Healthy
    # Vaccinate some individuals at t=0
    grid[0, rng.random((isize, jsize)) <= p_vax] = 1      # initial vaccination
    # Initial infections at t=0
    inf = rng.random((isize, jsize)) <= pignite          # initial infections
    # Vaccinated individuals cannot be infected at t=0
    inf[grid[0] == 1] = False
    if inf.sum() == 0:
        # If no infection occurred by chance, infect the center to start outbreak
        if grid[0, isize//2, jsize//2] == 2:
            inf[isize//2, jsize//2] = True
    grid[0, inf] = 3

    # Time stepping: synchronous update like the fire model
    for k in range(nstep - 1):
        grid[k+1] = grid[k]
        for i in range(isize):
            for j in range(jsize):
                # Only Sick individuals can infect neighbors or transition
                if grid[k, i, j] != 3:
                    continue
                # Try to infect healthy neighbors in 4 directions
                if i > 0 and grid[k, i-1, j] == 2 and rng.random() < pspread:
                    grid[k+1, i-1, j] = 3
                if i < isize-1 and grid[k, i+1, j] == 2 and rng.random() < pspread:
                    grid[k+1, i+1, j] = 3
                if j > 0 and grid[k, i, j-1] == 2 and rng.random() < pspread:
                    grid[k+1, i, j-1] = 3
                if j < jsize-1 and grid[k, i, j+1] == 2 and rng.random() < pspread:
                    grid[k+1, i, j+1] = 3
                # After spreading, Sick resolve into Immune or Dead
                if rng.random() < p_survive:
                    grid[k+1, i, j] = 1
                else:
                    grid[k+1, i, j] = 0
    return grid

def disease_time_series_plot(grid):
    """Single-run time series: %Sick / %Dead / %Immune / %Healthy.

    This produces the baseline epidemic curve used in the report.
    """
    ksize, isize, jsize = grid.shape
    n = isize * jsize
    # Count how many cells are in each health state at each time step
    sick   = [(grid[k] == 3).sum() for k in range(ksize)]
    dead   = [(grid[k] == 0).sum() for k in range(ksize)]
    immune = [(grid[k] == 1).sum() for k in range(ksize)]
    healthy = [(grid[k] == 2).sum() for k in range(ksize)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(range(ksize), pct(sick, n),   label="Sick",   color=COLOR_SICK)
    ax.plot(range(ksize), pct(dead, n),   label="Dead",   color=COLOR_DEAD)
    ax.plot(range(ksize), pct(immune, n), label="Immune", color=COLOR_IMMU)
    ax.plot(range(ksize), pct(healthy, n), label="Healthy", color=COLOR_HEAL)
    ax.set_title("Q3: Disease progression")
    ax.set_xlabel("Time step")
    ax.set_ylabel("% of population")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

def q3_sweep_psurvive_and_vax(isize=40, jsize=40, nstep=60,
                              pspread=0.6, pignite=0.02):
    """Two panels: (Peak Sick, Final Dead) vs p_survive; and vs p_vax.

    This function runs two separate sweeps:
      1) Vary p_survive with fixed vaccination and record peak sick and final dead.
      2) Vary p_vax with fixed survival probability and record the same diagnostics.
    These curves appear in the lab report to show how mortality
    and vaccination control outbreak severity.
    """
    # --------------------------------------------------------
    # Sweep survival probability p_survive
    # --------------------------------------------------------
    surv_vals = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    peak_sick, final_dead = [], []
    for psurv in surv_vals:
        g = disease_spread(isize, jsize, nstep,
                           pspread, pignite, psurv, p_vax=0.2)
        k, H, W = g.shape; n = H * W
        sick  = [(g[t] == 3).sum() for t in range(k)]
        dead  = [(g[t] == 0).sum() for t in range(k)]
        # Record diagnostics as percentages
        peak_sick.append(100.0 * max(sick) / n)
        final_dead.append(100.0 * dead[-1] / n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(surv_vals, peak_sick, marker='o')
    axes[0].set_title("Peak vs P_survive")
    axes[0].set_xlabel("P_survive")
    axes[0].set_ylabel("Peak % sick")
    axes[0].grid(alpha=0.25)

    axes[1].plot(surv_vals, final_dead, marker='o')
    axes[1].set_title("Deaths vs P_survive")
    axes[1].set_xlabel("P_survive")
    axes[1].set_ylabel("Final % dead")
    axes[1].grid(alpha=0.25)
    plt.show()

    # --------------------------------------------------------
    # Sweep vaccination fraction p_vax
    # --------------------------------------------------------
    vax_vals = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    peak_sick, final_dead = [], []
    for pv in vax_vals:
        g = disease_spread(isize, jsize, nstep,
                           pspread, pignite, p_survive=0.8, p_vax=pv)
        k, H, W = g.shape; n = H * W
        sick  = [(g[t] == 3).sum() for t in range(k)]
        dead  = [(g[t] == 0).sum() for t in range(k)]
        peak_sick.append(100.0 * max(sick) / n)
        final_dead.append(100.0 * dead[-1] / n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(vax_vals, peak_sick, marker='o')
    axes[0].set_title("Peak vs vaccination")
    axes[0].set_xlabel("p_vax")
    axes[0].set_ylabel("Peak % sick")
    axes[0].grid(alpha=0.25)

    axes[1].plot(vax_vals, final_dead, marker='o')
    axes[1].set_title("Deaths vs vaccination")
    axes[1].set_xlabel("p_vax")
    axes[1].set_ylabel("Final % dead")
    axes[1].grid(alpha=0.25)
    plt.show()


# ============================================================
# Run all functions 
# ============================================================
if __name__ == "__main__":

    # --- Q1 ---
    print("Running Q1: 3x3 snapshot demo...")
    q1_snapshots_3x3()

    print("Running Q1: 3x9 wide grid demo...")
    q1_wide_grid_demo()

    # --- Q2 ---
    print("Running Q2: Sweeping P_spread...")
    # Q2 experiment 1: vary P_spread
    q2_sweep_pspread(isize=30, jsize=30, nstep=35,
                       pignite=0.01, pbare=0.0, reps=8)
    
    print("Running Q2: Sweeping P_bare (density)...")
    # Q2 experiment 2: vary P_bare (initial non-forest)
    q2_sweep_pbare(isize=30, jsize=30, nstep=35,
                     pspread=0.6, pignite=0.05, reps=8)

    # --- Q3 ---
    print("Running Q3: Baseline time series...")
    g = disease_spread(isize=60, jsize=60, nstep=60,
                       pspread=0.6, pignite=0.02,
                       p_survive=0.8, p_vax=0.2)
    disease_time_series_plot(g)

    print("Running Q3: Sweeping P_survive and P_vax...")
    q3_sweep_psurvive_and_vax(isize=60, jsize=60, nstep=60,
                                pspread=0.6, pignite=0.02)
