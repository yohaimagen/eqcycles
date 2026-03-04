import numpy as np
import pandas as pd
import geopandas as gpd
import ot
from typing import Tuple, Dict, Any, Union, Optional
from joblib import Parallel, delayed

from eqcycles.core.data import SimulationData
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.analysis.rupture import get_rupture_mask


def prepare_event_data(
    catalog_df: pd.DataFrame, shapefile_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a geographic catalog into a 2D space-time representation for OT.

    Args:
        catalog_df: DataFrame with `time`, `lon_start`, `lat_start`, etc.
        shapefile_path: Path to the fault geometry shapefile.

    Returns:
        A tuple of (coords, masses):
        - coords: (N, 2) array of (along-strike location, time).
        - masses: (N,) array of rupture lengths.
    """
    if catalog_df.empty:
        return np.array([]).reshape(0, 2), np.array([])

    # Load the fault trace and project it to a suitable planar CRS for distance calcs
    fault_gdf = gpd.read_file(shapefile_path)
    if fault_gdf.empty:
        raise ValueError(f"Shapefile at {shapefile_path} is empty or could not be read.")
    
    # Use the UTM CRS appropriate for the fault's centroid
    planar_crs = fault_gdf.estimate_utm_crs()
    fault_line_planar = fault_gdf.to_crs(planar_crs).geometry.iloc[0]

    # Create GeoDataFrames for start and end points to project them
    start_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(catalog_df.lon_start, catalog_df.lat_start), crs="EPSG:4326"
    ).to_crs(planar_crs)
    
    end_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(catalog_df.lon_end, catalog_df.lat_end), crs="EPSG:4326"
    ).to_crs(planar_crs)

    # Project the planar start/end points onto the 1D fault line to get distances
    dist_start = np.array([fault_line_planar.project(point) for point in start_gdf.geometry])
    dist_end = np.array([fault_line_planar.project(point) for point in end_gdf.geometry])

    # Location is the minimum distance along strike (most "easterly" point, assuming trace starts East)
    location_m = np.minimum(dist_start, dist_end)
    # Mass is the rupture length
    mass_m = np.abs(dist_end - dist_start)

    # Filter out events with negligible mass to prevent numerical instability in OT
    valid_mask = mass_m > 1e-3
    location_m = location_m[valid_mask]
    mass_m = mass_m[valid_mask]
    time = catalog_df['time'].values[valid_mask]
    
    # Combine into the final (N, 2) coordinate matrix for OT
    coords = np.column_stack((location_m, time))
    
    return coords, mass_m


def prepare_sim_event_data(
    sim_data: SimulationData, 
    shapefile_path: str,
    mag_threshold: float = 7.0,
    rupture_threshold: float = 0.05,
    num_bins: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts event data from a SimulationData object for OT analysis.

    Args:
        sim_data: The SimulationData object.
        shapefile_path: Path to the fault geometry shapefile.
        mag_threshold: Minimum magnitude to include.
        rupture_threshold: Slip threshold (m) to define rupture extent.
        num_bins: Number of bins for spatial discretization.

    Returns:
        A tuple of (coords, masses):
        - coords: (N, 2) array of (along-strike location, time).
        - masses: (N,) array of rupture lengths.
    """
    mesh_along_strike = project_to_fault_trace(sim_data.coords, shapefile_path)
    
    coords = []
    masses = []
    
    for idx, event in sim_data.catalog.iterrows():
        if event.Mw < mag_threshold:
            continue
            
        centers, is_rup = get_rupture_mask(
            sim_data, 
            idx, 
            mesh_along_strike, 
            num_bins=num_bins, 
            slip_threshold=rupture_threshold
        )
        
        if not np.any(is_rup):
            continue
            
        rup_dists = centers[is_rup]
        min_dist = np.min(rup_dists)
        max_dist = np.max(rup_dists)
        rup_len = max_dist - min_dist

        # Only include events with non-negligible rupture length
        if rup_len > 1e-3:
            # Location is the minimum distance along strike (easternmost point)
            coords.append([min_dist, event.Time_year])
            # Mass is the rupture length
            masses.append(rup_len)
        
    if not coords:
        return np.array([]).reshape(0, 2), np.array([])
        
    return np.array(coords), np.array(masses)

def normalize_coords(
    coords: np.ndarray, scale_x: float, scale_t: float
) -> np.ndarray:
    """Applies scaling factors to space-time coordinates."""
    if coords.size == 0:
        return coords
    
    normalized = coords.copy()
    normalized[:, 0] /= scale_x
    normalized[:, 1] /= scale_t
    return normalized

def calculate_ot_score(
    coords1: np.ndarray, masses1: np.ndarray, 
    coords2: np.ndarray, masses2: np.ndarray, 
    config: Dict[str, Any]
) -> float:
    """
    Calculates the Unbalanced Optimal Transport distance between two point clouds
    with a weak penalty for topological sequence inversions.

    Args:
        coords1: (N, 2) array of normalized space-time coordinates for dataset 1.
        masses1: (N,) array of masses for dataset 1.
        coords2: (M, 2) array of normalized space-time coordinates for dataset 2.
        masses2: (M,) array of masses for dataset 2.
        config: Dictionary containing:
            - 'reg': Entropic regularization parameter for Sinkhorn.
            - 'reg_m': Marginal relaxation parameter for Unbalanced OT.
            - 'seq_weight': (Optional) Weight for the topological sequence penalty.
                A value > 0 penalizes backward temporal jumps in the matching.

    Returns:
        The OT distance (float).
    """
    if len(masses1) == 0 or len(masses2) == 0:
        return np.inf

    # Normalize masses to sum to 1.0 for numerical stability
    m1_sum = np.sum(masses1)
    m2_sum = np.sum(masses2)
    avg_total_mass = (m1_sum + m2_sum) / 2.0
    
    masses1_norm = masses1 / m1_sum
    masses2_norm = masses2 / m2_sum

    # Cost matrix: Euclidean distance in the normalized space-time plane
    cost_matrix = ot.dist(coords1, coords2, metric='euclidean')
    # Normalize cost matrix by its mean to keep 'reg' parameters stable
    mean_cost = np.mean(cost_matrix)
    if mean_cost > 1e-12:
        normalized_cost_matrix = cost_matrix / mean_cost
    else:
        normalized_cost_matrix = cost_matrix

    # Get transport plan P (coupling matrix)
    P = ot.unbalanced.sinkhorn_unbalanced(
        masses1_norm, masses2_norm, normalized_cost_matrix, 
        reg=config['reg'], reg_m=config['reg_m']
    )
    
    # Base OT score
    base_score = np.sum(P * normalized_cost_matrix)

    # Topological Sequence Penalty
    seq_weight = config.get('seq_weight', 0.0)
    sequence_penalty = 0.0
    
    if seq_weight > 0 and len(coords1) > 1 and len(coords2) > 1:
        # Array of simulation indices
        sim_indices = np.arange(len(coords2))
        
        # Calculate expected simulation index for each historical event
        row_sums = P.sum(axis=1)
        
        # Safeguard against division by zero
        safe_row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        expected_sim_indices = np.dot(P, sim_indices) / safe_row_sums
        
        # Backward jumps in the timeline (topological inversions)
        index_steps = np.diff(expected_sim_indices)
        inversions = np.maximum(0, -index_steps)
        sequence_penalty = seq_weight * np.sum(inversions)

    # Rescale the final score back to physical units (average mass units)
    return float((base_score + sequence_penalty) * avg_total_mass)

def get_transport_plan(
    coords1: np.ndarray, masses1: np.ndarray, 
    coords2: np.ndarray, masses2: np.ndarray, 
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Exposes the coupling matrix P (transport plan) for visualization.
    Matches the logic in calculate_ot_score exactly.

    Args:
        coords1: (N, 2) array of normalized space-time coordinates for dataset 1.
        masses1: (N,) array of masses for dataset 1.
        coords2: (M, 2) array of normalized space-time coordinates for dataset 2.
        masses2: (M,) array of masses for dataset 2.
        config: Dictionary containing 'reg' and 'reg_m' parameters.

    Returns:
        The (N, M) coupling matrix P.
    """
    if len(masses1) == 0 or len(masses2) == 0:
        return np.zeros((len(masses1), len(masses2)))

    # Normalize masses to sum to 1.0
    m1_sum = np.sum(masses1)
    m2_sum = np.sum(masses2)
    avg_total_mass = (m1_sum + m2_sum) / 2.0
    
    masses1_norm = masses1 / m1_sum
    masses2_norm = masses2 / m2_sum

    # Cost matrix calculation
    cost_matrix = ot.dist(coords1, coords2, metric='euclidean')
    mean_cost = np.mean(cost_matrix)
    if mean_cost > 1e-12:
        normalized_cost_matrix = cost_matrix / mean_cost
    else:
        normalized_cost_matrix = cost_matrix

    # Compute transport plan P
    P = ot.unbalanced.sinkhorn_unbalanced(
        masses1_norm, masses2_norm, normalized_cost_matrix, 
        reg=config['reg'], reg_m=config['reg_m']
    )
    
    # Rescale P by the physical units
    return P * avg_total_mass

def _process_single_window(
    t_start: float,
    t_end: float,
    window_edg: float,
    sim_times: np.ndarray,
    sim_coords: np.ndarray,
    sim_masses_norm: np.ndarray,
    norm_hist_coords: np.ndarray,
    hist_masses_norm: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[float, float]:
    """Helper for parallel window processing."""
    # Step 3: Fast slicing with searchsorted
    idx_start = np.searchsorted(sim_times, t_start - window_edg, side='left')
    idx_end = np.searchsorted(sim_times, t_end + window_edg, side='right')
    
    window_coords = sim_coords[idx_start:idx_end]
    window_masses_subset = sim_masses_norm[idx_start:idx_end]
    
    if window_coords.shape[0] == 0:
        return t_start, np.inf

    # Time-shift window coordinates to start at t_start
    relative_window_coords = window_coords.copy()
    relative_window_coords[:, 1] -= t_start

    # Normalize space and time
    norm_window_coords = normalize_coords(
        relative_window_coords, config['scale_x'], config['scale_t']
    )

    # Calculate score
    score = calculate_ot_score(
        norm_hist_coords, hist_masses_norm,
        norm_window_coords, window_masses_subset,
        config
    )
    return t_start, score

def find_best_sequence(
    hist_coords: np.ndarray,
    hist_masses: np.ndarray,
    sim_coords: np.ndarray,
    sim_masses: np.ndarray,
    config: Dict[str, Any],
    window_edg: float = 50.0,
    parallel_jobs: int = 10
) -> pd.DataFrame:
    """
    Finds the best match for a historical sequence in a simulation catalog
    using a sliding window, Optimal Transport, and parallelization.

    Args:
        hist_coords: (N, 2) array of (along-strike location, time) for historical events.
        hist_masses: (N,) array of masses (e.g., rupture lengths) for historical events.
        sim_coords: (M, 2) array of (along-strike location, time) for simulation events.
        sim_masses: (M,) array of masses for simulation events.
        config: Dictionary with OT parameters:
            - 'scale_x', 'scale_t', 'scale_mass': Normalization factors.
            - 'reg', 'reg_m': Sinkhorn parameters.
            - 'step_years': Sliding window step size.
            - 'seq_weight': (Optional) Weight for the topological sequence penalty.
        window_edg: Padding around the historical duration in the simulation window.
        parallel_jobs: Number of parallel jobs for window processing.

    Returns:
        A pandas DataFrame with columns ['time', 'score'] detailing the OT
        distance at each window position.
    """
    if hist_coords.size == 0:
        raise ValueError("Historical coordinates are empty.")
    
    if sim_coords.size == 0:
        print("Warning: Simulation coordinates are empty.")
        return pd.DataFrame(columns=['time', 'score'])

    # 1. Guarantee Chronological Sorting
    sort_idx = np.argsort(sim_coords[:, 1])
    sim_coords = sim_coords[sort_idx]
    sim_masses = sim_masses[sort_idx]
    sim_times = sim_coords[:, 1]

    # 2. Prepare Historical Data (Pre-calculate shared values)
    hist_duration = hist_coords[:, 1].max() - hist_coords[:, 1].min()
    
    relative_hist_coords = hist_coords.copy()
    relative_hist_coords[:, 1] -= hist_coords[:, 1].min()
    norm_hist_coords = normalize_coords(
        relative_hist_coords, config['scale_x'], config['scale_t']
    )
    
    hist_masses_norm = hist_masses / config['scale_mass']
    sim_masses_norm = sim_masses / config['scale_mass']

    # 3. Initialize Sliding Window
    sim_start_time = sim_times.min()
    sim_end_time = sim_times.max()
    step_years = config.get('step_years', 1.0)
    
    window_starts = np.arange(sim_start_time, sim_end_time - hist_duration, step_years)

    # 4. Joblib's Parallel Execution
    print(f"Scanning {len(window_starts)} windows with Joblib parallelization...")
    results = Parallel(n_jobs=parallel_jobs)(
        delayed(_process_single_window)(
            t_start, t_start + hist_duration, window_edg,
            sim_times, sim_coords, sim_masses_norm,
            norm_hist_coords, hist_masses_norm,
            config
        ) for t_start in window_starts
    )
    
    # 5. Reassemble and Return
    results_df = pd.DataFrame(results, columns=['time', 'score'])
    return results_df
