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
    code: int           # -1 for neg-unilateral, 0 for unclear, 1 for pos-unilateral, 2 for bilateral
    slope_pos: Optional[float] = None # Propagation velocity inverse (s/m) for positive-going branch
    r2_score_pos: Optional[float] = None
    slope_neg: Optional[float] = None # Propagation velocity inverse (s/m) for negative-going branch
    r2_score_neg: Optional[float] = None
    hypocenter_dist: Optional[float] = None # Along-strike distance of the hypocenter
    hypocenter_time: Optional[float] = None # Time of the hypocenter in seconds

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


def get_rupture_locations_and_times(
    sim_data: SimulationData, 
    event_idx: int, 
    mesh_along_strike: np.ndarray,
    rupture_threshold: float,
    pre_event_time: float,
    post_event_time: float,
    verbose: bool = False
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Retrieves the along-strike locations and corresponding rupture arrival times for a specific event.

    Args:
        sim_data: The standardized simulation data object.
        event_idx: The index of the earthquake in the catalog.
        mesh_along_strike: The along-strike distance for each mesh node.
        rupture_threshold: The slip rate (m/s) to define rupture arrival.
        pre_event_time: Time window in seconds before the event to consider.
        post_event_time: Time window in seconds after the event to consider.
        verbose: If True, print detailed analysis steps.

    Returns:
        A tuple (node_dists, arrival_times) if successful, otherwise None.
    """
    event_row = sim_data.catalog.iloc[event_idx]
    event_time_sec = event_row['Time_sec']
    
    # 1. Define Time Window based on event time
    t_start_sec = event_time_sec - pre_event_time
    t_end_sec = event_time_sec + post_event_time
    
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
    
    return node_dists, arrival_times


def analyze_rupture_direction(
    sim_data: SimulationData, 
    event_idx: int, 
    mesh_along_strike: np.ndarray,
    rupture_threshold: float = DEFAULT_RUPTURE_THRESHOLD,
    verbose: bool = False
) -> Optional[RuptureMetrics]:
    """
    Analyzes the rupture propagation direction for a specific event using linear regression.
    It tries to fit a single unilateral model first. If the fit is poor, it
    attempts to fit a bilateral model.

    Args:
        sim_data: The standardized simulation data object.
        event_idx: The index of the earthquake in the catalog.
        mesh_along_strike: The along-strike distance for each mesh node.
        rupture_threshold: The slip rate (m/s) to define rupture arrival.
        verbose: If True, print detailed analysis steps.

    Returns:
        A RuptureMetrics object containing the analysis results, or None if the
        analysis cannot be completed.
    """
    event_row = sim_data.catalog.iloc[event_idx]
    
    if verbose:
        event_time_sec = event_row['Time_sec']
        print(f"  Analyzing Dir: Event {event_idx} (Mw {event_row['Mw']:.1f}) at {event_time_sec:.2f}s")

    rupture_data = get_rupture_locations_and_times(
        sim_data,
        event_idx,
        mesh_along_strike,
        rupture_threshold,
        PRE_EVENT_TIME,
        POST_EVENT_TIME,
        verbose=verbose
    )

    if rupture_data is None:
        return RuptureMetrics(code=0)

    node_dists, arrival_times = rupture_data
    
    if len(node_dists) < 3:
        if verbose: print("    Not enough ruptured nodes to analyze direction.")
        return RuptureMetrics(code=0)

    # 4. Perform Linear Regression
    X = node_dists.reshape(-1, 1)
    y = arrival_times
    
    # First, try to fit a single line (unilateral model)
    reg_all = LinearRegression().fit(X, y)
    slope_all = reg_all.coef_[0]
    r2_all = reg_all.score(X, y)
    hypo_idx = np.argmin(arrival_times) 
    if r2_all > MIN_R2_THRESHOLD:
        
        
        hypo_dist = node_dists[hypo_idx]
        hypo_time = arrival_times[hypo_idx]

        if slope_all > 0:
            if verbose:
                velocity = 1 / (slope_all * 1000) if slope_all != 0 else float('inf')
                print(f"    Unilateral Positive Fit: Slope={slope_all:.2e} s/m, R2={r2_all:.2f}, Est. Vel={velocity:.2f} km/s")
            return RuptureMetrics(code=1, slope_pos=slope_all, r2_score_pos=r2_all, hypocenter_dist=hypo_dist, hypocenter_time=hypo_time)
        else:
            if verbose:
                velocity = 1 / (slope_all * 1000) if slope_all != 0 else float('inf')
                print(f"    Unilateral Negative Fit: Slope={slope_all:.2e} s/m, R2={r2_all:.2f}, Est. Vel={velocity:.2f} km/s")
            return RuptureMetrics(code=-1, slope_neg=slope_all, r2_score_neg=r2_all, hypocenter_dist=hypo_dist, hypocenter_time=hypo_time)

    # If unilateral fit is poor, try a bilateral model
    if verbose:
        print(f"    Unilateral fit poor (R2={r2_all:.2f}). Trying bilateral model.")
    
    # BILATERAL CASE: Define hypocenter as the point of earliest arrival time.
    hypocenter_idx = np.argmin(arrival_times)
    hypo_dist = node_dists[hypocenter_idx]
    hypo_time = arrival_times[hypocenter_idx]

    pos_branch_mask = node_dists >= hypo_dist
    neg_branch_mask = node_dists <= hypo_dist

    if np.sum(pos_branch_mask) < 3 or np.sum(neg_branch_mask) < 3:
        if verbose: print("    Not enough data points for bilateral fit.")
        return RuptureMetrics(code=0)

    X_pos, y_pos = node_dists[pos_branch_mask].reshape(-1, 1), arrival_times[pos_branch_mask]
    X_neg, y_neg = node_dists[neg_branch_mask].reshape(-1, 1), arrival_times[neg_branch_mask]

    reg_pos = LinearRegression().fit(X_pos, y_pos)
    slope_pos = reg_pos.coef_[0]
    r2_pos = reg_pos.score(X_pos, y_pos)

    reg_neg = LinearRegression().fit(X_neg, y_neg)
    slope_neg = reg_neg.coef_[0]
    r2_neg = reg_neg.score(X_neg, y_neg)

    is_pos_good = r2_pos > MIN_R2_THRESHOLD and slope_pos > 0
    is_neg_good = r2_neg > MIN_R2_THRESHOLD and slope_neg < 0

    if is_pos_good and is_neg_good:
        if verbose:
            vel_pos = 1 / (slope_pos * 1000) if slope_pos != 0 else float('inf')
            vel_neg = 1 / (slope_neg * 1000) if slope_neg != 0 else float('inf')
            print(f"    Bilateral Fit Success: ")
            print(f"      - Pos. Branch: R2={r2_pos:.2f}, Vel={vel_pos:.2f} km/s")
            print(f"      - Neg. Branch: R2={r2_neg:.2f}, Vel={vel_neg:.2f} km/s")
        return RuptureMetrics(code=2, slope_pos=slope_pos, r2_score_pos=r2_pos, slope_neg=slope_neg, r2_score_neg=r2_neg, hypocenter_dist=hypo_dist, hypocenter_time=hypo_time)

    if verbose:
        print(f"    Bilateral fit failed or inconclusive. R2_pos={r2_pos:.2f} (Slope={slope_pos:.2e}), R2_neg={r2_neg:.2f} (Slope={slope_neg:.2e})")

    return RuptureMetrics(code=0)
