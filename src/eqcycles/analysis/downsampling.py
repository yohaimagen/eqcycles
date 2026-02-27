import numpy as np
import meshio
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from eqcycles.core.data import SimulationData

def calculate_triangle_areas(mesh: meshio.Mesh) -> np.ndarray:
    """
    Calculates the surface area of each triangle cell in the mesh.
    
    Args:
        mesh: A meshio.Mesh object containing 'triangle' cells.
        
    Returns:
        np.ndarray: An array of areas for each triangle.
    """
    if "triangle" not in mesh.cells_dict:
        raise ValueError("Mesh does not contain 'triangle' cells.")
    
    triangles = mesh.cells_dict["triangle"]
    points = mesh.points
    
    v0 = points[triangles[:, 0]]
    v1 = points[triangles[:, 1]]
    v2 = points[triangles[:, 2]]
    
    # Area = 0.5 * |(v1 - v0) x (v2 - v0)|
    cross_product = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    return areas

def get_triangle_centroids(mesh: meshio.Mesh) -> np.ndarray:
    """
    Calculates the centroids of all triangle cells in the mesh.
    
    Args:
        mesh: A meshio.Mesh object.
        
    Returns:
        np.ndarray: (N, 3) array of centroids.
    """
    triangles = mesh.cells_dict["triangle"]
    points = mesh.points
    centroids = points[triangles].mean(axis=1)
    return centroids

def downsample_simulation(
    high_res_data: SimulationData, 
    target_mesh: Union[meshio.Mesh, str, Path],
    scale_factor: float = 1.2,
    z_limit: float = -18.0,
    output_dir: Optional[str] = None,
    run_id: str = "downsampled"
) -> SimulationData:
    """
    Downsamples simulation data from high-resolution to low-resolution.
    
    Args:
        high_res_data: The source SimulationData object.
        target_mesh: The target meshio.Mesh object.
        scale_factor: Multiplier for search radius (default 1.2).
        z_limit: Depth threshold for boundary filtering (default -18.0 km).
        output_dir: Optional directory to save the downsampled data.
        run_id: Suffix for saved files if output_dir is provided.
        
    Returns:
        SimulationData: A new SimulationData object with downsampled fields.
    """
    # 1. Geometry and Area Extraction
    if isinstance(target_mesh, (str, Path)):
        target_mesh = meshio.read(target_mesh)
    source_areas = calculate_triangle_areas(high_res_data.mesh)
    source_centroids = get_triangle_centroids(high_res_data.mesh)
    
    target_areas = calculate_triangle_areas(target_mesh)
    target_centroids = get_triangle_centroids(target_mesh)
    
    n_source = len(source_centroids)
    n_target = len(target_centroids)
    
    # 2. Spatial Indexing
    tree = cKDTree(source_centroids)
    
    # 3. Mapping Matrix Generation
    # weights[target_idx] = list of (source_idx, weight)
    # We use a weighted list for each target cell
    mapping_weights = []
    
    # Track source cell "claim counts" for normalization
    source_claim_counts = np.zeros(n_source)
    
    # Initial pass to find neighbors and claim counts
    target_neighbors = []
    for j in range(n_target):
        radius = scale_factor * np.sqrt(target_areas[j] / np.pi)
        indices = tree.query_ball_point(target_centroids[j], radius)
        
        # Boundary and Depth Filtering
        z_target = target_centroids[j, 2]
        # Filter source indices that cross the z_limit transition
        filtered_indices = []
        for idx in indices:
            z_source = source_centroids[idx, 2]
            # If target is above limit and source is below, or vice versa, skip
            if (z_target > z_limit and z_source < z_limit) or (z_target < z_limit and z_source > z_limit):
                continue
            filtered_indices.append(idx)
            
        # Fallback: Nearest Neighbor if filtered list is empty
        if not filtered_indices:
            _, idx = tree.query(target_centroids[j], k=1)
            filtered_indices = [idx]
            
        target_neighbors.append(filtered_indices)
        for idx in filtered_indices:
            source_claim_counts[idx] += 1

    # 4. Data Aggregation and Normalization
    def apply_mapping(field: np.ndarray) -> np.ndarray:
        if field is None:
            return None
        
        # field shape: (n_source, dim2) where dim2 is ntime or num_events
        # or (n_source,)
        if field.ndim == 1:
            target_field = np.zeros(n_target)
        else:
            target_field = np.zeros((n_target, field.shape[1]))
            
        for j in range(n_target):
            indices = target_neighbors[j]
            total_weighted_area = 0.0
            
            for idx in indices:
                # Weight = Neighbor_Area / claim_count
                weight = source_areas[idx] / source_claim_counts[idx]
                target_field[j] += weight * field[idx]
                total_weighted_area += weight
            
            if total_weighted_area > 0:
                target_field[j] /= total_weighted_area
                
        return target_field

    # Downsample all fields
    downsampled_slip = apply_mapping(high_res_data.slip)
    downsampled_shear_stress = apply_mapping(high_res_data.shear_stress)
    downsampled_normal_stress = apply_mapping(high_res_data.normal_stress)
    downsampled_state_variable = apply_mapping(high_res_data.state_variable)
    downsampled_slip_rate = apply_mapping(high_res_data.slip_rate)
    downsampled_eq_slip = apply_mapping(high_res_data.eq_slip)

    # 5. Catalog Mapping
    new_catalog = high_res_data.catalog.copy()
    if 'Hypo_Node' in new_catalog.columns:
        # Find closest target centroid for each Hypo_Node
        target_tree = cKDTree(target_centroids)
        # Assuming Hypo_Node matches source cell indices
        source_nodes = high_res_data.catalog['Hypo_Node'].values.astype(int)
        # Note: Some nodes might be vertices in original data, but SimulationData 
        # coords/cells usually refer to centroids in our analysis. 
        # If Hypo_Node refers to vertices, this needs adjustment. 
        # HBI loader sets Hypo_Node, but we'll assume they map to centroids for now.
        hypo_coords = source_centroids[source_nodes]
        _, target_nodes = target_tree.query(hypo_coords, k=1)
        new_catalog['Hypo_Node'] = target_nodes

    # 6. Construct new SimulationData
    # Prepare mesh_verts for plotting
    triangles = target_mesh.cells_dict["triangle"]
    mesh_verts = target_mesh.points[triangles]
    
    mesh_limits = [
        target_mesh.points[:,0].min(), target_mesh.points[:,0].max(),
        target_mesh.points[:,1].min(), target_mesh.points[:,1].max(),
        target_mesh.points[:,2].min(), target_mesh.points[:,2].max()
    ]

    result = SimulationData(
        time=high_res_data.time,
        coords=target_centroids,
        mesh=target_mesh,
        mesh_verts=mesh_verts,
        mesh_limits=mesh_limits,
        eq_slip=downsampled_eq_slip,
        catalog=new_catalog,
        slip_rate=downsampled_slip_rate,
        state_variable=downsampled_state_variable,
        shear_stress=downsampled_shear_stress,
        normal_stress=downsampled_normal_stress,
        slip=downsampled_slip
    )
    
    if output_dir:
        result.save(output_dir, run_id=run_id)
        
    return result
