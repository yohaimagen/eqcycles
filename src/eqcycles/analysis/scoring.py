import numpy as np
import pandas as pd
import geopandas as gpd
import ot
from typing import Tuple, Dict, Any

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

    time = catalog_df['time'].values
    
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
        
        # Location is the minimum distance along strike (easternmost point)
        coords.append([min_dist, event.Time_year])
        # Mass is the rupture length
        masses.append(max_dist - min_dist)
        
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
    Calculates the Unbalanced Optimal Transport distance between two point clouds.
    """
    # Cost matrix: Euclidean distance in the normalized space-time plane
    cost_matrix = ot.dist(coords1, coords2, metric='euclidean')

    score = ot.unbalanced.sinkhorn_unbalanced2(
        masses1, masses2, cost_matrix, 
        reg=config['reg'], reg_m=config['reg_m']
    )
    return score

from typing import Tuple, Dict, Any, Union, Optional

def find_best_sequence(
    hist_coords: np.ndarray,
    hist_masses: np.ndarray,
    sim_coords: np.ndarray,
    sim_masses: np.ndarray,
    config: Dict[str, Any],
    window_edg: float = 50.0
) -> pd.DataFrame:
    """
    Finds the best match for a historical sequence in a simulation catalog
    using a sliding window and Optimal Transport.

    Args:
        hist_coords: (N, 2) array of (along-strike location, time) for historical events.
        hist_masses: (N,) array of masses (e.g., rupture lengths) for historical events.
        sim_coords: (M, 2) array of (along-strike location, time) for simulation events.
        sim_masses: (M,) array of masses for simulation events.
        config: Dictionary with OT parameters (`scale_x`, `scale_t`, `scale_mass`, `reg`, `reg_m`, `step_years`).
        window_edg: Padding around the historical duration in the simulation window.

    Returns:
        A pandas DataFrame with columns ['time', 'score'] detailing the OT
        distance at each window position.
    """
    if hist_coords.size == 0:
        raise ValueError("Historical coordinates are empty.")
    
    if sim_coords.size == 0:
        print("Warning: Simulation coordinates are empty.")
        return pd.DataFrame(columns=['time', 'score'])

    # 1. Prepare Historical Data
    hist_duration = hist_coords[:, 1].max() - hist_coords[:, 1].min()
    
    # Normalize masses for stability
    hist_masses_norm = hist_masses / config['scale_mass']
    sim_masses_norm = sim_masses / config['scale_mass']

    # 2. Initialize Sliding Window
    sim_start_time = sim_coords[:, 1].min()
    sim_end_time = sim_coords[:, 1].max()
    step_years = config.get('step_years', 1.0)
    
    window_starts = np.arange(sim_start_time, sim_end_time - hist_duration, step_years)
    results = []

    # 3. Loop Through Windows
    print(f"Scanning {len(window_starts)} windows...")
    for t_start in window_starts:
        t_end = t_start + hist_duration

        # Create window subset
        window_mask = (sim_coords[:, 1] >= t_start - window_edg) & (sim_coords[:, 1] < t_end + window_edg)
        
        window_coords = sim_coords[window_mask]
        window_masses_subset = sim_masses_norm[window_mask]
        
        if window_coords.shape[0] == 0:
            results.append((t_start, np.inf))
            continue

        # Time-shift both sets of coordinates to start at t=0
        relative_hist_coords = hist_coords.copy()
        relative_hist_coords[:, 1] -= hist_coords[:, 1].min()
        
        relative_window_coords = window_coords.copy()
        relative_window_coords[:, 1] -= t_start

        # Normalize space and time for both
        norm_hist_coords = normalize_coords(relative_hist_coords, config['scale_x'], config['scale_t'])
        norm_window_coords = normalize_coords(relative_window_coords, config['scale_x'], config['scale_t'])

        # Calculate score
        score = calculate_ot_score(
            norm_hist_coords, hist_masses_norm,
            norm_window_coords, window_masses_subset,
            config
        )
        results.append((t_start, score))
    
    # 4. Finalize and Return
    results_df = pd.DataFrame(results, columns=['time', 'score'])
    return results_df
