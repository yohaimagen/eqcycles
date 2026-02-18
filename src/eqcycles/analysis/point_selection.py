import numpy as np
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING

from eqcycles.analysis.geometry import project_to_fault_trace

if TYPE_CHECKING:
    from eqcycles.core.data import SimulationData

def find_node_at_point(
    sim_data: "SimulationData",
    shapefile_path: str,
    target_dist_km: float,
    target_depth_km: float
) -> int:
    """
    Finds the index of the mesh node closest to a target coordinate specified
    by along-strike distance and depth.

    Args:
        sim_data (SimulationData): The loaded simulation data.
        shapefile_path (str): Path to the fault trace shapefile.
        target_dist_km (float): Target along-strike distance in kilometers.
        target_depth_km (float): Target depth in kilometers (positive down).

    Returns:
        int: The integer index of the closest node in the simulation mesh.
    """
    # 1. Get node coordinates in (along-strike, depth) space, in kilometers.
    # project_to_fault_trace returns distances in meters.
    node_dists_km = project_to_fault_trace(sim_data.coords, shapefile_path) / 1000.0
    
    # Assuming the Z-coordinate from sim_data.coords is elevation (km),
    # with Z=0 at the surface and negative values for depth.
    node_depths_km = -sim_data.coords[:, 2]
    
    # 2. Create the coordinate array for the KD-Tree.
    node_coords_2d_km = np.column_stack((node_dists_km, node_depths_km))
    
    # 3. Find the closest node.
    tree = cKDTree(node_coords_2d_km)
    
    target_coords_km = np.array([target_dist_km, target_depth_km])
    
    _, closest_node_index = tree.query(target_coords_km, k=1)
    
    return int(closest_node_index)
