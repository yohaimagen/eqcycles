from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

from eqcycles.core.data import SimulationData

# --- Configuration Constants ---
# These were previously in the main script body. It's better to have them here.
PRE_EVENT_TIME = 24 * 60 * 60    # seconds (Window before event)
POST_EVENT_TIME = 24 * 60 * 60   # seconds (Window after event)
MIN_R2_THRESHOLD = 0.3           # R-squared threshold to declare "clear direction"
DEFAULT_RUPTURE_THRESHOLD = 0.05 # Default slip rate threshold (m/s) for rupture arrival

@dataclass
class RuptureMetrics:
    """
    Stores the calculated metrics for a single rupture event's propagation.
    """
    code: int           # -1 for negative strike, 0 for unclear, 1 for positive strike
    slope: float        # Propagation velocity inverse (s/m) from linear regression
    r2_score: float     # R-squared value of the linear regression fit

def get_rupture_mask(
    sim_data: SimulationData, 
    event_idx: int, 
    node_distances: np.ndarray, 
    num_bins: int, 
    slip_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    
    max_dist = np.max(node_distances)
    bin_edges = np.linspace(0, max_dist, num_bins + 1)
    
    # 1. Get 0-based bin indices for every node
    # We subtract 1 because digitize returns 1-based indices (0 is underflow)
    bin_indices = np.digitize(node_distances, bin_edges) - 1
    
    # Handle edge case: values exactly equal to max_dist will fall into the next bin index
    # We clip them to ensure they stay in the last valid bin (index num_bins - 1)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # 2. Get slip data
    slip_data = sim_data.eq_slip[:, event_idx]
    
    # 3. Compute max slip per bin using vectorization
    # Initialize with 0.0 (assuming slip is non-negative)
    max_slip_per_bin = np.zeros(num_bins)
    
    # np.maximum.at does the aggregation in a single compiled pass
    np.maximum.at(max_slip_per_bin, bin_indices, slip_data)
    
    # 4. Generate mask
    is_ruptured = max_slip_per_bin > slip_threshold
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, is_ruptured


def analyze_rupture_direction(
    sim_data: SimulationData, 
    event_idx: int, 
    mesh_along_strike: np.ndarray,
    rupture_threshold: float = DEFAULT_RUPTURE_THRESHOLD,
    verbose: bool = False
) -> Optional[RuptureMetrics]:
    """
    Analyzes the rupture propagation direction for a specific event using linear regression.

    Args:
        sim_data: The standardized simulation data object.
        event_idx: The index of the earthquake in the catalog.
        mesh_along_strike: The along-strike distance for each mesh node.
        rupture_threshold: The slip rate (m/s) to define rupture arrival.
        verbose: If True, print detailed analysis steps.

    Returns:
        A RuptureMetrics object containing the analysis results, or None if the
        analysis cannot be completed (e.g., insufficient data).
    """
    event_row = sim_data.catalog.iloc[event_idx]
    event_time_sec = event_row['Time_sec']
    
    if verbose:
        print(f"  Analyzing Dir: Event {event_idx} (Mw {event_row['Mw']:.1f}) at {event_time_sec:.2f}s")

    # 1. Define Time Window based on event time
    t_start_sec = event_time_sec - PRE_EVENT_TIME
    t_end_sec = event_time_sec + POST_EVENT_TIME
    
    # The stored time vector is in years, convert to seconds for comparison
    t_sim_sec = sim_data.time * (365 * 24 * 60 * 60)
    
    # Find the indices in the simulation's time vector that bound our window
    time_mask = (t_sim_sec >= t_start_sec) & (t_sim_sec <= t_end_sec)
    
    if np.sum(time_mask) < 2:
        if verbose: print("    Warning: Not enough time steps in the analysis window.")
        return None

    # 2. Extract Data for the analysis window
    # The slip_rate is log10(vel), so convert back to linear velocity
    sr_window_linear = 10**sim_data.slip_rate[:, time_mask]
    time_window_sec = t_sim_sec[time_mask]
    
    # 3. Find rupture arrival times at each node
    is_rupturing = sr_window_linear > rupture_threshold
    
    # Find all nodes that ruptured at any point during the window
    nodes_that_ruptured_mask = np.any(is_rupturing, axis=1)
    
    if not np.any(nodes_that_ruptured_mask):
        if verbose: print("    Warning: No nodes exceeded rupture threshold in the window.")
        return None

    # Filter down to only the nodes that ruptured
    ruptured_nodes_indices = np.where(nodes_that_ruptured_mask)[0]
    rupture_timing_matrix = is_rupturing[nodes_that_ruptured_mask, :]
    
    # For each ruptured node, find the index of the *first* time step it ruptured
    first_rupture_time_indices = np.argmax(rupture_timing_matrix, axis=1)
    
    # Get the actual arrival times and corresponding distances
    arrival_times = time_window_sec[first_rupture_time_indices]
    node_dists = mesh_along_strike[ruptured_nodes_indices]
    
    # 4. Perform Linear Regression: time = f(distance)
    X = node_dists.reshape(-1, 1)
    y = arrival_times
    
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]  # This is dt/dx, the inverse of velocity
    r2 = reg.score(X, y)
    
    # 5. Determine direction code based on slope and R-squared
    direction_code = 0  # Default to 'unclear'
    if r2 > MIN_R2_THRESHOLD:
        if slope > 0:
            direction_code = 1   # Positive Strike (propagating towards increasing distance)
        else:
            direction_code = -1  # Negative Strike (propagating towards decreasing distance)
            
    if verbose:
        velocity = 1 / (slope * 1000) if slope != 0 else float('inf')
        print(f"    Slope={slope:.2e} s/m, R2={r2:.2f}, Code={direction_code}, Est. Vel={velocity:.2f} km/s")
    
    return RuptureMetrics(code=direction_code, slope=slope, r2_score=r2)
