#!/usr/bin/env python3

'''
Final Project — Integrated Script (Q1-Q5)

This single file collects all code needed for:
  • Q1 — GMPE implementation and baseline attenuation validation
  • Q2 — Magnitude-sensitivity analysis of damage radius
  • Q3 — 2-D ShakeMaps and empirical site-amplification factor
  • Q4 — Probabilistic hazard assessment with Monte Carlo sampling
  • Q5 — 1-D FDTD elastic-wave solver for physics-based site amplification

How to run everything
---------------------
$ python3 project.py
    - Generates the full set of figures for Q1-Q5 in sequence.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# Project Settings
# ==========================================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'figure.figsize': (10, 6),
    'figure.dpi': 120
})

print("Initialization: Libraries loaded and settings configured.\n")

# ==============================================================================
# Part 1: The Empirical Statistical Engine
# Implementation of GMPE
# ==============================================================================

class BooreGMPE:
    """
    Implementation of the Boore, Joyner, & Fumal (1997) Ground Motion
    Prediction Equation (GMPE).

    This class provides:
      - predict():  returns PGA for given Mw, distance, and Vs30
      - pga_to_intensity(): converts PGA to Modified Mercalli Intensity (MMI)

    The formulation here is simplified but follows the structure of the
    original regression: a magnitude term, a distance (path) term, and
    a site term, plus an optional aleatory-uncertainty perturbation.
    Predicts Peak Ground Acceleration (PGA).
    """
    def __init__(self):
        # Regression Coefficients 
        # (values chosen to give realistic attenuation and site effects)
        self.b1_ss = -0.313
        self.b2 = 0.527
        self.b3 = 0.000      # Usually 0 for M < 8.5
        self.b5 = -0.778
        self.bv = -0.371
        self.Va = 1396.0     # Reference Shear Wave Velocity (m/s)
        self.h = 5.57        # Fictitious depth term (km)
        
        # Standard Deviation (Aleatory Uncertainty) - used for Monte Carlo
        self.sigma_lnY = 0.52 

    def predict(self, Mw, R_jb, Vs30, epsilon=0):
        """
        Calculates PGA (g) from the GMPE for given inputs.

        Args:
            Mw (float): Moment Magnitude of the earthquake.
            R_jb (float or array): Joyner-Boore distance (km). Can be a scalar
                or a NumPy array to enable vectorized evaluation.
            Vs30 (float): Shear wave velocity averaged over top 30m (m/s).
            epsilon (float or array): Standard normal random variable
                representing aleatory uncertainty. For deterministic runs,
                use the default value of 0. For Monte Carlo, pass an array.

        Returns:
            numpy.ndarray or float: Predicted Peak Ground Acceleration (g).
               Shape matches the broadcasted shapes of R_jb and epsilon.
        """
        # Distance Term Correction: R = sqrt(R_jb^2 + h^2)
        #    This regularizes the near-source behavior so we never divide by 0.
        R = np.sqrt(R_jb**2 + self.h**2)
        
        # Magnitude Scaling Term
        F_mag = self.b1_ss + self.b2 * (Mw - 6) + self.b3 * (Mw - 6)**2
        
        # Distance Attenuation Term (Geometric Spreading + attenuation)
        F_dist = self.b5 * np.log(R)
        
        # Site Amplification Term: compares Vs30 to a reference rock velocity.
        F_site = self.bv * np.log(Vs30 / self.Va)
        
        # Combine terms to get ln(Y)
        # Includes optional aleatory uncertainty term: epsilon * sigma.
        ln_Y = F_mag + F_dist + F_site + (epsilon * self.sigma_lnY)
        
        # Return PGA in g (log space is natural log, so use np.exp).
        return np.exp(ln_Y)

    def pga_to_intensity(self, pga):
        """
        Converts PGA to Modified Mercalli Intensity (MMI) based on the
        piecewise relationship in Wald et al. (1999).

        Args:
            pga (float or ndarray): Peak ground acceleration in units of g.

        Returns:
            ndarray or float: Estimated MMI values clipped to [I, X].
        """
        # Avoid log(0) error by enforcing a tiny floor.
        pga = np.maximum(pga, 1e-6)
        
        mmi = np.zeros_like(pga)
        # Wald's piecewise function 
        mask_low = pga < 0.1
        mask_high = pga >= 0.1
        
        # Low intensity regime
        mmi[mask_low] = 1.0 + 3.66 * np.log10(pga[mask_low]) + 1.66
        # High intensity regime
        mmi[mask_high] = 5.0 + 3.66 * np.log10(pga[mask_high])
        
        # Clip range between I and X so we stay in the standard MMI range.
        return np.clip(mmi, 1, 10)


# ==============================================================================
# Part 2: The Physics Engine
# Implementation of 1D Finite-Difference Time-Domain (FDTD) Wave Solver
# ==============================================================================

class Wave1DSolver:
    """
    1D Elastic Wave Equation FDTD Solver.

    Solves the vertical shear-wave problem
        rho(z) * u_tt = d/dz [ mu(z) * du/dz ]
    on a uniform grid using an explicit leapfrog scheme and a simple
    stress-free upper boundary.

    Typical use:
        solver = Wave1DSolver(depth=100, nz=100)
        solver.set_material('soil')  # or 'rock'
        t, input_trace, surface_trace = solver.solve(input_amplitude=0.1)
    """
    def __init__(self, depth=100, nz=100, total_time=2.0):
        # Domain and grid
        self.H = depth         # Total depth (m)
        self.nz = nz           # Number of grid points
        self.dz = depth / nz   # Spatial step size (m)
        self.T = total_time    # Total simulation time (s)
        
        # Material Property Arrays (functions of depth)
        self.rho = np.zeros(nz) # Density (kg/m3)
        self.mu = np.zeros(nz)  # Shear Modulus (Pa)
        self.vs = np.zeros(nz)  # Shear Velocity (m/s)
        
        # Wavefield State Arrays
        self.u = np.zeros(nz)      # Displacement at current step u^n
        self.u_prev = np.zeros(nz) # Displacement at previous step u^{n-1}
        self.u_next = np.zeros(nz) # Displacement at next step u^{n+1}
        
        # Grid Coordinates (Index 0 is surface, Index -1 is bedrock bottom)
        self.z_coords = np.linspace(0, depth, nz)

    def set_material(self, material_type='rock'):
        """
        Configures the geological material profile for the column.

        Args:
            material_type (str): 'rock' for a homogeneous hard-rock column,
                or 'soil' for a soft-soil layer (top 30 m) over hard rock.
        """
        if material_type == 'rock':
            # Homogeneous Hard Rock
            self.vs[:] = 760.0   # m/s
            self.rho[:] = 2500.0 # kg/m3
        elif material_type == 'soil':
            # Soft Soil Layer over Rock
            # Deep rock baseline
            self.vs[:] = 760.0
            self.rho[:] = 2500.0
            
            # Top 30m is soft soil
            soil_layers = int(30 / self.dz)
            self.vs[0:soil_layers] = 200.0   # Soft soil velocity
            self.rho[0:soil_layers] = 1800.0 # Soft soil density
            
        # Calculate Shear Modulus: mu = rho * vs^2
        self.mu = self.rho * self.vs**2

    def get_ricker_wavelet(self, t, freq=5.0, amplitude=1.0):
        """
        Generates a Ricker wavelet source time function.

        Args:
            t (1D array): time axis.
            freq (float): dominant frequency of the wavelet (Hz).
            amplitude (float): overall amplitude scale.

        Returns:
            1D array of the same length as t containing the wavelet.
        """
        t0 = 1.5 / freq # Peak delay so the wavelet is centered away from t=0.
        arg = (np.pi * freq * (t - t0))**2
        return amplitude * (1 - 2 * arg) * np.exp(-arg)

    def solve(self, input_amplitude=0.01, record_field=False):
        """
        Executes the FDTD simulation loop.

        The bottom boundary is driven by a prescribed displacement
        (Ricker wavelet). The top boundary is a free surface (zero
        shear stress). Optionally, the full wavefield can be recorded.

        Args:
            input_amplitude (float): peak amplitude of the source wavelet.
            record_field (bool): if True, store u(z,t) for all times.

        Returns:
            t_axis (1D array): time axis.
            input_record (1D array): displacement at the bottom node.
            surface_record (1D array): displacement at the surface node.
            full_field (2D array, optional): u(t,z) if record_field=True.
        """
        # 1. Stability Check (CFL Condition)
        # dt <= dz / max_vs
        max_vs = np.max(self.vs)
        dt_crit = self.dz / max_vs
        dt = dt_crit * 0.9  # Safety factor of 0.9 to stay below the CFL limit
        
        nt = int(self.T / dt) # Number of time steps
        t_axis = np.arange(nt) * dt
        
        print(f"  [Physics Solver] Config: dz={self.dz}m, dt={dt:.5f}s (CFL Safe), nt={nt}")
        
        # 2. Prepare Source (Input from bottom)
        # The source is a Ricker wavelet scaled by the desired amplitude.
        source_wave = self.get_ricker_wavelet(t_axis, freq=8.0, amplitude=input_amplitude)
        
        # 3. Initialize Recorders
        surface_record = np.zeros(nt) # Record at surface
        input_record = np.zeros(nt)   # Record at input depth
        full_field = np.zeros((nt, self.nz)) if record_field else None
        
        # 4. Time Stepping Loop
        # Discrete Equation: u_next = 2u - u_prev + (dt^2/rho) * (stress_divergence)
        for n in range(nt):
            # Boundary Condition: Forced Input at Bottom
            self.u[-1] = source_wave[n]
            
            # Update Interior Points
            # Calculate discrete stress gradient: d/dz(mu * du/dz)
            for i in range(1, self.nz - 1):
                # Variable coefficient central difference for mu * du/dz
                mu_plus = 0.5 * (self.mu[i+1] + self.mu[i])
                mu_minus = 0.5 * (self.mu[i] + self.mu[i-1])
                
                term1 = mu_plus * (self.u[i+1] - self.u[i])
                term2 = mu_minus * (self.u[i] - self.u[i-1])
                
                div_stress = (term1 - term2) / (self.dz**2)
                acceleration = div_stress / self.rho[i]
                
                self.u_next[i] = 2*self.u[i] - self.u_prev[i] + acceleration * (dt**2)
            
            # Boundary Condition: Free Surface at Top
            # Stress = 0 -> du/dz = 0 -> u[0] = u[1]
            self.u_next[0] = self.u_next[1]
            
            # Update state for next step (shift time levels)
            self.u_prev[:] = self.u[:]
            self.u[:] = self.u_next[:]
            
            # Record data
            surface_record[n] = self.u[0]
            input_record[n] = self.u[-1]
            if record_field:
                full_field[n, :] = self.u[:]
            
        if record_field:
            return t_axis, input_record, surface_record, full_field
        else:
            return t_axis, input_record, surface_record


# ==============================================================================
# PART 3: Execution Blocks 
# ==============================================================================

def plot_q1_validation():
    """
    Q1: Baseline Validation.

    Generates an attenuation curve for an M7.0 event on bedrock using
    the GMPE and plots PGA versus distance on log-log axes. The plot
    is used as a visual validation that the geometric spreading and
    near-field behavior look reasonable.
    """
    print("\n--- Running Q1: Baseline Validation ---")
    gmpe_model = BooreGMPE()
    distances = np.logspace(0, 2, 100) # 1km to 100km
    
    # Predict for M7.0 on Bedrock
    pga = gmpe_model.predict(Mw=7.0, R_jb=distances, Vs30=760)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(distances, pga, 'b-', lw=3, label='Mw 7.0 (Bedrock)')
    plt.axhline(0.1, color='gray', ls='--', label='Damage Threshold (0.1g)')
    
    plt.title('Fig 1: Baseline Validation (Attenuation Curve)')
    plt.xlabel('Distance (km)')
    plt.ylabel('Peak Ground Acceleration (g)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def plot_q2_sensitivity():
    """
    Q2: Magnitude Sensitivity Analysis.

    Uses the same GMPE to compare attenuation curves for Mw 5, 6, 7,
    and 8 on bedrock. For each magnitude, the function also prints an
    approximate "damage radius" where PGA drops below 0.1 g.
    """
    print("\n--- Running Q2: Magnitude Sensitivity ---")
    gmpe_model = BooreGMPE()
    distances = np.logspace(0, 2, 100)
    magnitudes = [5.0, 6.0, 7.0, 8.0]
    colors = ['green', 'blue', 'orange', 'red']
    
    plt.figure(figsize=(8, 6))
    
    for Mw, col in zip(magnitudes, colors):
        pga = gmpe_model.predict(Mw, distances, Vs30=760)
        plt.loglog(distances, pga, lw=2, color=col, label=f'Mw {Mw}')
        
        # Calculate Damage Radius for this magnitude.
        idx = np.where(pga > 0.1)[0]
        if len(idx) > 0:
            radius = distances[idx[-1]]
            print(f"  Mw {Mw}: Damage Radius (>0.1g) ~ {radius:.1f} km")
        else:
            print(f"  Mw {Mw}: Damage Radius ~ 0 km")

    plt.axhline(0.1, color='k', ls='--', label='Damage Threshold (0.1g)')
    plt.title('Fig 2: Magnitude Sensitivity Analysis')
    plt.xlabel('Distance (km)')
    plt.ylabel('Peak Ground Acceleration (g)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def generate_shakemaps():
    """
    Q3: 2D Site Amplification ShakeMaps.

    Builds a 100 x 100 km grid around the epicenter and evaluates the
    GMPE for an M7.0 event for two site conditions:
      - hard rock (Vs30 = 760 m/s)
      - soft soil (Vs30 = 200 m/s)

    The function converts PGA to MMI and plots side-by-side ShakeMaps.
    It also prints an empirical site amplification factor at a fixed
    distance for comparison with the physics-based result.
    """
    print("\n--- Running Q3: 2D Site Amplification ---")
    gmpe_model = BooreGMPE()
    
    # Create 100x100 km Grid centered on the epicenter at (0, 0).
    grid_res = 100
    x = np.linspace(-50, 50, grid_res)
    y = np.linspace(-50, 50, grid_res)
    X, Y = np.meshgrid(x, y)
    R_jb = np.sqrt(X**2 + Y**2) # Epicenter at (0,0)
    
    # Scenario A: Hard Rock
    pga_rock = gmpe_model.predict(Mw=7.0, R_jb=R_jb, Vs30=760)
    mmi_rock = gmpe_model.pga_to_intensity(pga_rock)
    
    # Scenario B: Soft Soil
    pga_soil = gmpe_model.predict(Mw=7.0, R_jb=R_jb, Vs30=200)
    mmi_soil = gmpe_model.pga_to_intensity(pga_soil)
    
    # Calculate Empirical Amplification Factor at ~10km distance
    dist_idx = int(grid_res/2) + 10 
    emp_amp_factor = pga_soil[dist_idx, dist_idx] / pga_rock[dist_idx, dist_idx]
    print(f"  Predicted Empirical Amplification Factor: {emp_amp_factor:.2f}x")
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    cmap = 'Reds'
    levels = np.linspace(2, 9, 15) # MMI II to IX
    
    c1 = ax[0].contourf(X, Y, mmi_rock, levels=levels, cmap=cmap)
    ax[0].set_title('Fig 3A: Scenario A (Hard Rock, Vs30=760)')
    ax[0].set_xlabel('X (km)')
    ax[0].set_ylabel('Y (km)')
    fig.colorbar(c1, ax=ax[0], label='MMI Intensity')
    
    c2 = ax[1].contourf(X, Y, mmi_soil, levels=levels, cmap=cmap)
    ax[1].set_title('Fig 3B: Scenario B (Soft Soil, Vs30=200)')
    ax[1].set_xlabel('X (km)')
    fig.colorbar(c2, ax=ax[1], label='MMI Intensity')
    
    plt.suptitle('Q3: 2D ShakeMaps Comparison (M7.0 Event)', fontsize=14)
    plt.tight_layout()
    plt.show()

def run_monte_carlo():
    """
    Q4: Probabilistic Risk Assessment (Monte Carlo).

    Samples the lognormal aleatory uncertainty in the GMPE for a
    single soft-soil site (Mw 7.0, R = 20 km, Vs30 = 200 m/s).
    The function generates a histogram of PGA, and overlays the
    median and 90th-percentile values.
    """
    print("\n--- Running Q4: Monte Carlo Risk Assessment ---")
    gmpe_model = BooreGMPE()
    
    # Define Scenario: M7.0, 20km, Soft Soil
    target_dist = 20.0
    target_vs30 = 200.0
    n_sims = 2000
    
    # Generate random noise (epsilon) for the log-space residuals.
    epsilon = np.random.normal(0, 1, n_sims)
    
    # Batch prediction in one vectorized call.
    mc_pgas = gmpe_model.predict(7.0, target_dist, target_vs30, epsilon)
    
    # Statistics
    median_pga = np.median(mc_pgas)
    p90_pga = np.percentile(mc_pgas, 90)
    
    print(f"  Median PGA: {median_pga:.3f} g")
    print(f"  90th Percentile PGA: {p90_pga:.3f} g (Worst Case)")
    
    plt.figure(figsize=(8, 5))
    plt.hist(mc_pgas, bins=40, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    plt.axvline(median_pga, color='b', lw=3, label=f'Median ({median_pga:.2f}g)')
    plt.axvline(p90_pga, color='r', ls='--', lw=3, label=f'90th %ile ({p90_pga:.2f}g)')
    
    plt.title('Fig 4: Probabilistic Hazard Distribution (Monte Carlo)')
    plt.xlabel('PGA (g)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def compute_pga_from_displacement(trace, dt):
    """
    Compute PGA (peak ground acceleration) from a displacement time series.

    Args:
        trace (1D array): displacement time series u(t).
        dt (float): time step between samples (seconds).

    Returns:
        float: peak ground acceleration (same length units per s^2 as u).

    Notes:
        We approximate derivatives with centered finite differences using
        numpy.gradient, which is sufficient for diagnostic comparison of
        rock vs. soil amplification in this project.
    """
    # First derivative: velocity
    vel = np.gradient(trace, dt)
    # Second derivative: acceleration
    acc = np.gradient(vel, dt)
    # Peak ground acceleration (PGA)
    return np.max(np.abs(acc))

def run_physics_comparison():
    """
    Q5: Physics-Based Verification (1D Wave Equation).

    Uses the Wave1DSolver to compare a homogeneous rock column and a
    soft-soil-over-rock column. Both are driven with the same Ricker
    wavelet scaled to the bedrock PGA from the GMPE. The function
    reports surface PGA for each case and plots the time series and
    a depth-time diagram of the soft-soil wavefield.
    """
    print("\n--- Running Q5: Physics-Based Verification ---")
    gmpe_model = BooreGMPE()
    
    # 1. Get Input Motion (Bedrock PGA from GMPE)
    bedrock_input_pga = gmpe_model.predict(7.0, 20.0, 760.0)
    print(f"  Step 1: Calculated Bedrock Input PGA = {bedrock_input_pga:.3f} g")
    
    # 2. Run Simulation for Rock
    print("  Step 2: Running FDTD Simulation for Rock site...")
    solver_rock = Wave1DSolver(depth=100, nz=100)
    solver_rock.set_material('rock')
    t, in_rock, out_rock = solver_rock.solve(input_amplitude=bedrock_input_pga)
    
    # 3. Run Simulation for Soft Soil
    print("  Step 3: Running FDTD Simulation for Soft Soil site...")
    solver_soil = Wave1DSolver(depth=100, nz=100)
    solver_soil.set_material('soil') # Top 30m is soft
    t_soil, in_soil, out_soil, field_soil = solver_soil.solve(input_amplitude=bedrock_input_pga,
                                                              record_field=True)
    
    # 4. Calculate Physical Amplification based on PGA
    dt = t[1] - t[0]

    # PGA at input and surface for both sites
    pga_in_rock   = compute_pga_from_displacement(in_rock, dt)
    pga_out_rock  = compute_pga_from_displacement(out_rock, dt)
    pga_in_soil   = compute_pga_from_displacement(in_soil, dt)
    pga_out_soil  = compute_pga_from_displacement(out_soil, dt)

    # Surface PGA ratio between soft-soil and rock sites (to compare with A_emp)
    phys_amp_factor = pga_out_soil / pga_out_rock

    print(f"  PGA (input, rock): {pga_in_rock:.3e}")
    print(f"  PGA (surface, rock): {pga_out_rock:.3e}")
    print(f"  PGA (surface, soft soil): {pga_out_soil:.3e}")
    print(f"  Physical Amplification Factor (surface PGA, soil/rock): {phys_amp_factor:.2f}x")
    
    # Plotting: time series
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, out_rock, 'b-', label='Surface (Hard Rock)')
    plt.plot(t, in_rock, 'k--', alpha=0.5, label='Input (Bedrock)')
    plt.title('Fig 5A: Physical Simulation - Hard Rock Site')
    plt.ylabel('Displacement')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, out_soil, 'r-', label='Surface (Soft Soil)')
    plt.plot(t, in_soil, 'k--', alpha=0.5, label='Input (Bedrock)')
    plt.title(f'Fig 5B: Physical Simulation - Soft Soil Site (Amp Factor = {phys_amp_factor:.2f}x)')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Plotting: space–time wavefield in soft-soil-over-rock column
    plt.figure(figsize=(6, 6))
    plt.imshow(field_soil.T,
               extent=[t_soil[0], t_soil[-1], solver_soil.H, 0],
               aspect='auto', cmap='seismic')
    plt.colorbar(label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (m)')
    plt.title('Fig 5C: Wavefield in Soft-Soil-over-Rock Column')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Ready to Run
# ==============================================================================
if __name__ == "__main__":
    plot_q1_validation()
    plot_q2_sensitivity()
    generate_shakemaps()
    run_monte_carlo()
    run_physics_comparison()
