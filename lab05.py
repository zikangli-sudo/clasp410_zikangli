#!/usr/bin/env python3

'''
Lab 5: Snowball Earth - Energy Balance Model and Stability Analysis.
- Core Physics Components:
  * Diffusion: Meridional heat transport (Implicit solver)
  * Spherical Correction: Accounting for latitudinal area changes
  * Radiative Forcing: Insolation (Shortwave) - Blackbody Cooling (Longwave)
  * Albedo Feedback: Dynamic switching (0.3 <-> 0.6) at T = -10°C

This script implements the full numerical pipeline used in the lab:
Q1: Solver validation (Diffusion only -> +Spherical -> +Radiative).
Q2: Parameter tuning (finding best lambda & epsilon to match Warm Earth).
Q3: Stability analysis using extreme initial conditions (Hot vs. Cold vs. Flash-freeze).
Q4: Hysteresis loop analysis (Varying solar multiplier gamma to find tipping points).

How to run everything
---------------------
$ python3 lab05.py
    - Generates the full set of figures for Q1-Q4 in sequence.  
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(npoints=18):
    '''
    Create a evenly spaced latitudinal grid with `npoints` cell centers.
    Grid will always run from zero to 180 as the edges of the grid. This
    means that the first grid point will be `dLat/2` from 0 degrees and the
    last point will be `180 - dLat/2`.

    Parameters
    ----------
    npoints : int, defaults to 18
        Number of grid points to create.

    Returns
    -------
    dLat : float
        Grid spacing in latitude (degrees)
    lats : numpy array
        Locations of all grid cell centers.
    '''

    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''
    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370):
    '''
    Solve the snowball Earth problem.

    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells.
    tfinal : int or float, defaults to 10,000
        Time length of simulation in years.
    dt : int or float, defaults to 1.0
        Size of timestep in years.
    lam : float, defaults to 100
        Set ocean diffusivity
    emiss : float, defaults to 1.0
        Set emissivity of Earth/ground.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat. Otherwise, the given values are used as-is.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : float, defaults to .6 and .3
        Set albedo values for ice and ground.

    Returns
    --------
    lats : Numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        180 is north.
    Temp : Numpy array
        Temperature as a function of latitude.
    '''

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = insolation(solar, lats)

    # Create temp array; set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix:
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create L matrix.
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):
        # Update Albedo:
        loc_ice = Temp <= -10  # Sea water freezes at ten below.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term:
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp


def problem1():
    '''
    Create solution figure for Problem 1 (also validate our code qualitatively)
    '''

    # Get warm Earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms:
    lats, temp_diff = snowball_earth()
    lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=.3)

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='Basic Diffusion')
    ax.plot(lats-90, temp_sphe, label='Diff. + Spherical Correction')
    ax.plot(lats-90, temp_alls, label='Diff. + SphCorr + Radiative')

    # Customize like those annoying insurance commercials
    ax.set_title('Final Steady State')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')


def test_functions():
    '''Test our functions'''

    print('Test gen_grid')
    print('For npoints=5:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    result = gen_grid(5)
    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {gen_grid(5)}")

# =============================================================================
# NEW CODE FOR PROBLEM 2 (Appended after starter code)
# =============================================================================

def problem2():
    '''
    Step 2: Tune the model using wide parameter ranges.
    lam (Diffusivity) range: 0 to 150
    emiss (Emissivity) range: 0 to 1
    '''
    print("--- Running Problem 2: Tuning Parameters ---")
    
    # Get target data (modern warm Earth)
    dlat, lats = gen_grid()
    target_temp = temp_warm(lats)
    
    # Set up the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # -----------------------------------------------------
    # Plot 1: Varying Diffusivity (lam) from 0 to 150
    # -----------------------------------------------------
    # Note: To see the shape clearly, we must fix emissivity to a 
    # reasonable value (e.g., 0.75) instead of 1.0, otherwise all 
    # curves will be too cold to compare effectively.
    fixed_emiss_test = 0.75 
    lam_values = [0, 50, 100, 150] # Range requested
    
    ax1.plot(lats-90, target_temp, 'k--', linewidth=2, label='Target (Warm Earth)')
    
    for val_lam in lam_values:
        # Full physics enabled (sphere + insol), albice fixed at 0.3 for tuning
        _, temp_res = snowball_earth(lam=val_lam, emiss=fixed_emiss_test, 
                                     apply_spherecorr=True, apply_insol=True,
                                     albice=0.3, albgnd=0.3)
        ax1.plot(lats-90, temp_res, label=f'lam={val_lam}')
        
    ax1.set_title(f'Effect of Diffusivity (fixed emiss={fixed_emiss_test})')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Latitude')
    ax1.legend()

    # -----------------------------------------------------
    # Plot 2: Varying Emissivity (emiss) from 0 to 1
    # -----------------------------------------------------
    # Pick a reasonable lambda (e.g., 30) to see the vertical shifts
    fixed_lam_test = 30
    emiss_values = [0.4, 0.6, 0.8, 1.0] # Range requested (0.0 is too hot to plot nicely)
    
    ax2.plot(lats-90, target_temp, 'k--', linewidth=2, label='Target (Warm Earth)')
    
    for val_emiss in emiss_values:
        _, temp_res = snowball_earth(lam=fixed_lam_test, emiss=val_emiss, 
                                     apply_spherecorr=True, apply_insol=True,
                                     albice=0.3, albgnd=0.3)
        ax2.plot(lats-90, temp_res, label=f'emiss={val_emiss}')

    ax2.set_title(f'Effect of Emissivity (fixed lam={fixed_lam_test})')
    ax2.set_xlabel('Latitude')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()



def find_absolute_minimum():
    '''
    Automatic Solver:
    Scans a dense grid of parameters to find the absolute lowest RMSE.
    Ranges based on your previous plots:
    - lam: 20 to 50 (Scanning for higher diffusivity)
    - emiss: 0.68 to 0.78 (Scanning for stronger greenhouse)
    '''
    print("\n--- Running Automatic Optimizer... Please wait ---")
    
    # 1. Get Target Data
    dlat, lats = gen_grid()
    target_temp = temp_warm(lats)
    
    # 2. Define Search Grid (Dense Scan)
    # lam: check every integer from 20 to 50
    lam_range = np.arange(20, 51, 1) 
    # emiss: check every 0.005 from 0.68 to 0.78
    emiss_range = np.arange(0.68, 0.78, 0.005)
    
    best_rmse = 9999.0
    best_lam = -1
    best_emiss = -1
    
    # 3. Brute Force Search
    total_combinations = len(lam_range) * len(emiss_range)
    count = 0
    
    for lam in lam_range:
        for emiss in emiss_range:
            # Run model
            _, model_temp = snowball_earth(lam=lam, emiss=emiss, 
                                           apply_spherecorr=True, apply_insol=True,
                                           albice=0.3, albgnd=0.3)
            # Calculate Error
            rmse = np.sqrt(np.mean((model_temp - target_temp)**2))
            
            # Save if best
            if rmse < best_rmse:
                best_rmse = rmse
                best_lam = lam
                best_emiss = emiss
            
            # Progress bar 
            count += 1
            if count % 100 == 0:
                print(f"Scanned {count}/{total_combinations} combinations...", end='\r')

    print(f"\n\n====== OPTIMIZATION COMPLETE ======")
    print(f"Absolute Best Parameters Found:")
    print(f" >> Diffusivity (lam) : {best_lam}")
    print(f" >> Emissivity (emiss): {best_emiss:.3f}")
    print(f" >> Minimum Error     : {best_rmse:.4f} °C")
    print(f"===================================")
    
    # 4. Plot the Winner
    _, final_temp = snowball_earth(lam=best_lam, emiss=best_emiss, 
                                   apply_spherecorr=True, apply_insol=True,
                                   albice=0.3, albgnd=0.3)
    
    plt.figure(figsize=(8, 6))
    plt.plot(lats-90, target_temp, 'k--', linewidth=2, label='Target (Warm Earth)')
    plt.plot(lats-90, final_temp, 'r-', linewidth=3, label=f'Best Fit (RMSE={best_rmse:.2f})')
    plt.title(f'Optimized Model (lam={best_lam}, eps={best_emiss:.3f})')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()


# =============================================================================
# PROBLEM 3: Initial Conditions & Stability
# =============================================================================

def problem3():
    '''
    Problem 3: Explore stability by changing initial conditions.
    Using BEST parameters from Step 2: lam=32, emiss=0.715
    '''
    print("\n--- Running Problem 3 (Stability Analysis) ---")
    
    # 1. Set best parameters
    # -------------------------------------------
    BEST_LAM = 32.0
    BEST_EMISS = 0.715
    print(f"Using Tuned Parameters: lam={BEST_LAM}, emiss={BEST_EMISS}")
    
    dlat, lats = gen_grid()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 2. Case 1: Hot Earth Start (60°C)
    # -------------------------------------------
    # Initial condition: 60 degrees everywhere
    init_hot = np.ones(lats.size) * 60.0
    
    # Run simulation (must enable all physics terms)
    _, temp_hot = snowball_earth(lam=BEST_LAM, emiss=BEST_EMISS, 
                                 init_cond=init_hot, 
                                 apply_spherecorr=True, apply_insol=True,
                                 albice=0.6, albgnd=0.3) # Normal albedo
    
    ax.plot(lats-90, temp_hot, 'r-', linewidth=3, label='Start: Hot (60°C) -> Warm Earth')
    
    # Save this "Warm Earth" result for use in Case 3
    warm_earth_profile = temp_hot

    # 3. Case 2: Cold Earth Start (-60°C)
    # -------------------------------------------
    # Initial condition: -60 degrees everywhere
    init_cold = np.ones(lats.size) * -60.0
    
    _, temp_cold = snowball_earth(lam=BEST_LAM, emiss=BEST_EMISS, 
                                  init_cond=init_cold, 
                                  apply_spherecorr=True, apply_insol=True,
                                  albice=0.6, albgnd=0.3) # Normal albedo
    
    ax.plot(lats-90, temp_cold, 'b-', linewidth=3, label='Start: Cold (-60°C) -> Snowball')

    # 4. Case 3: Flash Freeze
    # -------------------------------------------
    # Requirement: Start from Warm Earth profile, but force albedo = 0.6
    print("Simulating Flash Freeze...")
    
    # Set albgnd to 0.6 so albedo is 0.6 regardless of ice presence
    _, temp_flash = snowball_earth(lam=BEST_LAM, emiss=BEST_EMISS,
                                   init_cond=warm_earth_profile, # Start: Warm Earth temperature
                                   albgnd=0.6, albice=0.6,       # Force high albedo (Flash Freeze)
                                   apply_spherecorr=True, apply_insol=True)

    ax.plot(lats-90, temp_flash, 'g--', linewidth=2, label='Start: Warm but Albedo=0.6')

    # 5. Plot decoration
    # -------------------------------------------
    ax.set_title(f'Stability Analysis in Different Initial Conditions')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Latitude')
    ax.axhline(-10, color='gray', linestyle=':', alpha=0.5, label='Freezing Point (-10°C)')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# PROBLEM 4: Hysteresis Loop (Solar Forcing)
# =============================================================================

def problem4():
    '''
    Problem 4: The Hysteresis Loop.
    Vary solar multiplier (gamma) from 1.4 down to 0.4, then back up to 1.4.
    Plot Average Global Temperature vs. Gamma.
    '''
    print("\n--- Running Problem 4 (Hysteresis Loop) ---")
    print("This will take a moment (running many simulations)...")

    # 1. Use best parameters found in Step 2
    # ---------------------------------------------------------
    BEST_LAM = 32.0       # <--- Ensure this matches your previous Best Value
    BEST_EMISS = 0.715    # <--- Ensure this matches your previous Best Value
    
    # 2. Set the sequence of solar multiplier (Gamma) changes
    # ---------------------------------------------------------
    # Requirement: From 0.4 to 1.4.
    # To draw the full loop, run in two phases:
    # Phase 1: Cooling process (Start from Warm Earth, decrease solar, until frozen)
    # Start gamma at 1.4 and decrease to 0.4
    gamma_down = np.arange(1.4, 0.39, -0.05) 
    
    # Phase 2: Warming process (Start from Snowball Earth, increase solar, until melted)
    # Start gamma at 0.4 and increase to 1.4
    gamma_up = np.arange(0.4, 1.41, 0.05)

    # 3. Prepare data recording
    # ---------------------------------------------------------
    gammas_plot_down = [] # Record cooling gamma
    temps_plot_down = []  # Record cooling avg temp
    
    gammas_plot_up = []   # Record warming gamma
    temps_plot_up = []    # Record warming avg temp

    dlat, lats = gen_grid()
    
    # Calculate weights for global average temp (Equator area large, poles small)
    # Weighted average is more scientifically accurate
    weights = np.cos(np.deg2rad(lats-90)) # Simple cosine weighting
    weights /= weights.sum()

    # 4. Phase 1: Cooling process (Cooling Leg)
    # ---------------------------------------------------------
    # Initial state: Hot Earth (Ensure starting as Warm Earth)
    current_temp = np.ones(lats.size) * 60.0 
    
    print("Running Cooling Phase (Red Line)...")
    for g in gamma_down:
        # Solar = 1370 * g
        _, current_temp = snowball_earth(lam=BEST_LAM, emiss=BEST_EMISS,
                                         init_cond=current_temp, # Critical: Use result from previous step as next initial condition!
                                         solar=1370*g,           # Change solar intensity
                                         apply_spherecorr=True, apply_insol=True,
                                         albice=0.6, albgnd=0.3)
        
        # Calculate weighted average temperature
        avg_t = np.sum(current_temp * weights)
        
        gammas_plot_down.append(g)
        temps_plot_down.append(avg_t)

    # 5. Phase 2: Warming process (Warming Leg)
    # ---------------------------------------------------------
    # Critical: current_temp is the state at end of Phase 1 (Snowball)
    # Continue using it for the next run
    
    print("Running Warming Phase (Blue Line)...")
    for g in gamma_up:
        _, current_temp = snowball_earth(lam=BEST_LAM, emiss=BEST_EMISS,
                                         init_cond=current_temp, # Inherit previous temperature (Snowball state)
                                         solar=1370*g,
                                         apply_spherecorr=True, apply_insol=True,
                                         albice=0.6, albgnd=0.3)
        
        avg_t = np.sum(current_temp * weights)
        
        gammas_plot_up.append(g)
        temps_plot_up.append(avg_t)

    # 6. Plotting (The Hysteresis Loop)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 7))
    
    # Plot cooling curve (Red, pointing left)
    plt.plot(gammas_plot_down, temps_plot_down, 'r-o', label='Cooling (Start Warm)')
    # Plot warming curve (Blue, pointing right)
    plt.plot(gammas_plot_up, temps_plot_up, 'b-o', label='Warming (Start Cold)')
    
    # Mark current solar (gamma=1.0)
    plt.axvline(1.0, color='k', linestyle='--', label='Current Solar (1.0)')

    # Decoration
    plt.title(f'Bistability of Earth Climate')
    plt.xlabel('Solar Multiplier (gamma)')
    plt.ylabel('Global Average Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    problem1() 
    problem2()
    find_absolute_minimum()
    problem3()
    problem4()