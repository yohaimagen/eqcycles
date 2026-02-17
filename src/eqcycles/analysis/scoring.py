import numpy as np
import pandas as pd
import geopandas as gpd
import ot
from typing import Tuple, Dict, Any

from eqcycles.core.data import SimulationData
from eqcycles.analysis.geometry import project_to_fault_trace


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
    
    fault_line_geo = fault_gdf.geometry.iloc[0]
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

    # Location is the maximum distance along strike (most "easterly" point)
    location_m = np.maximum(dist_start, dist_end)
    # Mass is the rupture length
    mass_m = np.abs(dist_end - dist_start)

    time = catalog_df['time'].values
    
    # Combine into the final (N, 2) coordinate matrix for OT
    coords = np.column_stack((location_m, time))
    
    return coords, mass_m

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

def find_best_sequence(
    sim_catalog: pd.DataFrame, 
    historical_catalog: pd.DataFrame, 
    shapefile_path: str, 
    window_edg: float,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Finds the best match for a historical sequence in a simulation catalog
    using a sliding window and Optimal Transport.

    Args:
        sim_catalog: DataFrame of the long simulation.
        historical_catalog: DataFrame of the target historical sequence.
        shapefile_path: Path to the fault geometry shapefile.
        config: Dictionary with OT parameters (`scale_x`, `scale_t`, `scale_mass`, `reg`, `reg_m`, `step_years`).

    Returns:
        A pandas DataFrame with columns ['time', 'score'] detailing the OT
        distance at each window position.
    """
    # 1. Prepare Historical Data
    hist_coords, hist_masses = prepare_event_data(historical_catalog, shapefile_path)
    if hist_coords.size == 0:
        raise ValueError("Historical catalog is empty or could not be processed.")
    hist_duration = hist_coords[:, 1].max() - hist_coords[:, 1].min()
    
    hist_masses /= hist_masses / config['scale_mass']
    
    # 2. Prepare Simulation Data
    sim_coords, sim_masses = prepare_event_data(sim_catalog, shapefile_path)
    if sim_coords.size == 0:
        print("Warning: Simulation catalog is empty.")
        return pd.DataFrame(columns=['time', 'score'])
    
    sim_masses /= config['scale_mass']

    # 3. Initialize Sliding Window
    sim_start_time = sim_coords[:, 1].min()
    sim_end_time = sim_coords[:, 1].max()
    step_years = config.get('step_years', 1.0)
    
    window_starts = np.arange(sim_start_time, sim_end_time - hist_duration, step_years)
    results = []

    # 4. Loop Through Windows
    print(f"Scanning {len(window_starts)} windows...")
    for t_start in window_starts:
        t_end = t_start + hist_duration

        # Create window subset
        window_mask = (sim_coords[:, 1] >= t_start - window_edg) & (sim_coords[:, 1] < t_end + window_edg)
        
        window_coords = sim_coords[window_mask]
        window_masses = sim_masses[window_mask]
        

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
            norm_hist_coords, hist_masses,
            norm_window_coords, window_masses,
            config
        )
        results.append((t_start, score))
    
    # 5. Finalize and Return
    results_df = pd.DataFrame(results, columns=['time', 'score'])
    return results_df
