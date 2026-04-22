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
    
    # Build the KD-Tree from the reference trace, scaling it to KILOMETERS
    # to match the units of the input sim_coords.
    trace_points_km = np.column_stack((trace_x * 1e-3, trace_y * 1e-3))
    tree = cKDTree(trace_points_km)

    # Query the tree to find the nearest reference trace point.
    # The input sim_coords are expected to be in KILOMETERS.
    # We only use the X and Y coordinates.
    sim_coords_2d_km = sim_coords[:, :2]
    distances, indices = tree.query(sim_coords_2d_km, k=1)
    
    # The result is the along-strike distance from interp_dist (which is in METERS)
    # corresponding to the closest point found in the km-based tree.
    node_along_strike = interp_dist[indices]
    
    return node_along_strike

def compute_along_strike_profile(
    mesh_along_strike: np.ndarray,
    values: np.ndarray,
    coords: np.ndarray,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
) -> tuple:
    """
    Aggregates node-level values into along-strike bins with a depth filter.

    Nodes deeper than ``max_depth_km`` are excluded.  Within each bin the
    remaining shallow-node values are averaged across depth.  Works for both
    scalar fields ``(n_nodes,)`` and multi-column fields ``(n_nodes, N)``.

    Args:
        mesh_along_strike: Along-strike distance (metres) for each mesh node,
            as returned by ``project_to_fault_trace``.
        values: Per-node scalar ``(n_nodes,)`` or multi-column array
            ``(n_nodes, N)`` to aggregate.
        coords: Mesh node coordinates ``(n_nodes, 3)``.  The absolute value of
            ``coords[:, 2]`` is used as depth in km.
        num_bins: Number of along-strike bins.
        max_depth_km: Only nodes with ``|coords[:, 2]| <= max_depth_km`` are
            included in the aggregation.

    Returns:
        bin_centers_km : ``(num_bins,)`` — bin centre positions; negative values
            follow the west-is-negative convention used in other plots.
        profile        : ``(num_bins,)`` or ``(num_bins, N)`` — depth-averaged,
            bin-averaged values.  Bins with no nodes are left as zero.
        valid          : ``(num_bins,)`` bool — True where at least one node
            contributed to the bin.
    """
    shallow_mask = np.abs(coords[:, 2]) <= max_depth_km
    if not np.any(shallow_mask):
        raise ValueError(
            f"No nodes found shallower than {max_depth_km} km. "
            "Check the depth convention in coords[:, 2]."
        )

    along_strike_shallow = mesh_along_strike[shallow_mask]
    values_shallow = values[shallow_mask]

    max_dist = float(mesh_along_strike.max())
    bin_edges = np.linspace(0, max_dist, num_bins + 1)
    bin_centers_km = -(bin_edges[:-1] + bin_edges[1:]) / 2 * 1e-3

    bin_idx = np.clip(np.digitize(along_strike_shallow, bin_edges) - 1, 0, num_bins - 1)

    count_per_bin = np.zeros(num_bins, dtype=int)
    np.add.at(count_per_bin, bin_idx, 1)
    valid = count_per_bin > 0

    if values_shallow.ndim == 1:
        profile = np.zeros(num_bins)
        np.add.at(profile, bin_idx, values_shallow)
        profile[valid] /= count_per_bin[valid]
    else:
        n_cols = values_shallow.shape[1]
        profile = np.zeros((num_bins, n_cols))
        for col in range(n_cols):
            np.add.at(profile[:, col], bin_idx, values_shallow[:, col])
        profile[valid] /= count_per_bin[valid, np.newaxis]

    return bin_centers_km, profile, valid


def get_geometry_context(shapefile_path: str):
    """
    Helper to get the reference line geometry and CRS.
    """
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    gdf = gpd.read_file(shapefile_path)
    original_crs = gdf.crs
    line_geom = gdf.to_crs(3857).geometry.iloc[0]
    
    return line_geom, original_crs
