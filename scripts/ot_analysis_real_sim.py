# %% [markdown]
# Optimal Transport Analysis: Historical 20th Century vs. Simulation
# This script performs a sliding window OT analysis to find the best match 
# for the historical NAF 20th-century rupture sequence in a real simulation.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import os

# autoreload
%load_ext autoreload
%autoreload 2

from eqcycles.io import HBILoader
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.vis.rupture_sequence_matplotlib import plot_rupture_sequence_matplotlib
from eqcycles.analysis.scoring import prepare_event_data, find_best_sequence, normalize_coords, calculate_ot_score

# %% [markdown]
# ## 1. Setup Paths and Configuration
# Adjust these paths as necessary for your local environment.

# %%
# TODO: Update these paths to match your local setup
MSH_PATH = './NAF_res5.msh' 
SIM_DATA_DIR = 'downsampled_output'
TRACE_SHAPEFILE = 'scripts/shapefiles/NAF_simplefied.shp'
HISTORICAL_CSV = 'historical_ruptures_20th_century.csv'
SIM_NUM = 13

# OT Configuration
config = {
    'scale_x': 100_000.0,    # Normalize space (meters) to ~100km units
    'scale_t': 100.0,        # Normalize time (years) to century units
    'scale_mass': 10_000.0,  # Normalize rupture length (meters)
    'reg': 0.1,              # Entropic regularization parameter
    'reg_m': 1.0,            # Marginal relaxation for unbalanced OT
    'step_years': 5.0        # Sliding window step size
}

window_edge = 50.0  # Padding around the historical duration in the simulation window

# %% [markdown]
# ## 2. Load Data

# %%
# Load Historical Catalog
hist_df = pd.read_csv(HISTORICAL_CSV)
# prepare_event_data expects a 'time' column
hist_df['time'] = hist_df['year']
print(f"Loaded {len(hist_df)} historical events.")

# Load Simulation Data
loader = HBILoader(MSH_PATH)
try:
    sim_data = loader.load(SIM_DATA_DIR, SIM_NUM, fields_to_load=['slip_rate'])
    print(f"Loaded Simulation {SIM_NUM}: {len(sim_data.catalog)} events.")
except Exception as e:
    print(f"Error loading simulation {SIM_NUM}: {e}")
    # Placeholder for script robustness
    sim_data = None

# %% [markdown]
# ## 3. Perform Sliding Window OT Scoring

# %%
if sim_data is not None:
    print("Running sliding window OT...")
    results = find_best_sequence(
        historical_catalog=hist_df,
        shapefile_path=TRACE_SHAPEFILE,
        sim_data=sim_data,
        window_edg=window_edge,
        config=config
    )
    
    # Save results
    results.to_csv(f'ot_scores_sim_{SIM_NUM}.csv', index=False)
    print("OT Analysis complete.")

# %% [markdown]
# ## 4. Visualize Results

# %%
if sim_data is not None and not results.empty:
    # 1. Plot Score vs. Time Shift
    plt.figure(figsize=(10, 4))
    plt.plot(results['time'], results['score'], label='OT Score')
    
    best_fit_idx = results['score'].idxmin()
    best_fit_time = results.loc[best_fit_idx, 'time']
    best_score = results.loc[best_fit_idx, 'score']
    
    plt.scatter(best_fit_time, best_score, color='red', zorder=5)
    plt.annotate(f'Best Fit: {best_fit_time:.1f} yr Score: {best_score:.4f}',
                 (best_fit_time, best_score), xytext=(10, 10), 
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    
    plt.xlabel('Simulation Time Shift (years)')
    plt.ylabel('Unbalanced OT Score')
    plt.title(f'Historical Sequence Match in Sim {SIM_NUM}')
    plt.grid(True, alpha=0.3)
    plt.show()

# %% [markdown]
# ## 5. Detailed OT Mapping for Best Fit

# %%
def get_transport_plan(coords1, masses1, coords2, masses2, config):
    import ot
    cost_matrix = ot.dist(coords1, coords2, metric='euclidean')
    return ot.unbalanced.sinkhorn_unbalanced(
        masses1, masses2, cost_matrix, reg=config['reg'], reg_m=config['reg_m']
    )

if sim_data is not None:
    from eqcycles.analysis.scoring import prepare_sim_event_data
    # Extract Best Window
    hist_duration = hist_df['time'].max() - hist_df['time'].min()
    t_start = best_fit_time
    t_end = t_start + hist_duration
    
    # Create a subset SimulationData for the window
    sim_window_data = sim_data.subset_time(t_start - window_edge, t_end + window_edge)
    
    # Prepare Data for Mapping
    hist_coords_raw, hist_masses_raw = prepare_event_data(hist_df, TRACE_SHAPEFILE)
    sim_coords_raw, sim_masses_raw = prepare_sim_event_data(sim_window_data, TRACE_SHAPEFILE)
    
    # Normalize
    hist_coords_norm = normalize_coords(hist_coords_raw, config['scale_x'], config['scale_t'])
    hist_coords_norm[:, 1] -= hist_coords_norm[:, 1].min() # Relative time
    hist_masses_norm = hist_masses_raw / config['scale_mass']
    
    sim_coords_norm = normalize_coords(sim_coords_raw, config['scale_x'], config['scale_t'])
    # Re-normalize sim time to be relative to the window start for OT
    sim_coords_norm[:, 1] = (sim_window_data.catalog['Time_year'].values - t_start) / config['scale_t']
    sim_masses_norm = sim_masses_raw / config['scale_mass']
    
    # Transport Plan
    P = get_transport_plan(hist_coords_norm, hist_masses_norm, sim_coords_norm, sim_masses_norm, config)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Project historical and simulation points to along-strike km for plotting
    hist_strike_km = hist_coords_raw[:, 0] * 1e-3
    sim_strike_km = sim_coords_raw[:, 0] * 1e-3
    
    # Plot Simulation Events
    for i, row in sim_window_data.catalog.iterrows():
        # Note: We use the actual simulation time for the Y axis
        ax.hlines(row['Time_year'], sim_strike_km[i] - 20, 
                  sim_strike_km[i] + 20, 
                  color='gray', alpha=0.5, linewidth=4)

    # Plot Historical Events (Shifted to Best Fit Time)
    for i, row in hist_df.iterrows():
        t_visual = (row['time'] - hist_df['time'].min()) + t_start
        ax.hlines(t_visual, hist_strike_km[i] - 20, hist_strike_km[i] + 20, 
                  color='black', linewidth=2)
        ax.text(hist_strike_km[i], t_visual + 2, str(row['year']), ha='center', fontsize=8)

    # Draw OT Connections
    threshold = 1e-3 * P.max()
    segments = []
    weights = []
    for i in range(len(hist_df)):
        for j in range(len(sim_window)):
            if P[i, j] > threshold:
                t_hist = (hist_df.iloc[i]['time'] - hist_df['time'].min()) + t_start
                t_sim = sim_window.iloc[j]['time']
                segments.append([(hist_strike_km[i], t_hist), (sim_strike_km[j], t_sim)])
                weights.append(P[i, j])
    
    if segments:
        weights = np.array(weights)
        norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())
        lc = LineCollection(segments, cmap='cool', norm=norm, alpha=0.6, linestyle='--')
        lc.set_array(weights)
        ax.add_collection(lc)
        plt.colorbar(lc, label='Mass Transported')

    ax.set_xlabel('Along-Strike Distance [km]')
    ax.set_ylabel('Time [years]')
    ax.set_title(f'OT Mapping: Historical vs. Sim {SIM_NUM} (Best Fit)')
    plt.grid(True, alpha=0.3)
    plt.show()
