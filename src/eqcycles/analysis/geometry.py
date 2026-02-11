import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
import os

def project_to_fault_trace(sim_coords: np.ndarray, shapefile_path: str) -> np.ndarray:
    """
    Projects simulation coordinates onto a fault trace defined by a shapefile.

    This function combines loading a reference trace and mapping mesh points to it.

    Args:
        sim_coords (np.ndarray): Array of simulation mesh node coordinates (Nx3).
        shapefile_path (str): Path to the shapefile defining the fault trace.

    Returns:
        np.ndarray: A 1D array of distances along the fault strike for each node.
    
    Raises:
        FileNotFoundError: If the shapefile is not found.
    """
    # 1. Get Reference Trace (formerly get_reference_trace)
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    naf_gdf = gpd.read_file(shapefile_path)
    # Project to a Cartesian CRS for distance calculations (Web Mercator)
    naf_cart = naf_gdf.to_crs(3857) 
    
    # Assuming the first geometry is the one we want
    line_geom = naf_cart.geometry.iloc[0]
    x_orig = np.array(line_geom.coords)[:, 0]
    y_orig = np.array(line_geom.coords)[:, 1]

    # Normalize coordinates to start at (0,0) for stable interpolation
    x_orig -= x_orig[0]
    y_orig -= y_orig[0]
    
    # Calculate cumulative distance along the reference trace
    dists = np.sqrt(np.diff(x_orig)**2 + np.diff(y_orig)**2)
    ref_dist = np.concatenate(([0], np.cumsum(dists)))

    # 2. Map Mesh to Strike (formerly map_mesh_to_strike)
    total_len = ref_dist[-1]
    spacing = 1000.0  # Interpolate the reference trace to a regular 1km spacing
    
    # Create a dense, regularly spaced representation of the fault trace
    interp_dist = np.arange(0, total_len, spacing)
    trace_x = np.interp(interp_dist, ref_dist, x_orig)
    trace_y = np.interp(interp_dist, ref_dist, y_orig)
    
    # The reference trace is 2D (X, Y), so we only use X and Y from sim_coords
    # We create a KD-Tree from the densified reference trace points for fast lookup
    # Note: The original code used 3D points with Z=0, which is also fine,
    # but since we're projecting 3D fault data onto a 2D map trace, a 2D tree is sufficient.
    trace_points_2d = np.column_stack((trace_x * 1e-3, trace_y * 1e-3, np.zeros_like(trace_x)))  # Convert to km and add Z=0 for consistency
    tree = cKDTree(trace_points_2d)
    
    # Query the tree to find the nearest reference trace point for each simulation node.
    # We only use the X and Y coordinates of the simulation mesh for this projection.
    distances, indices = tree.query(sim_coords[:, :3], k=1)
    
    # The result is the along-strike distance of the *closest reference point*
    # for each simulation node.
    node_along_strike = interp_dist[indices]
    
    return node_along_strike * 1e-3
