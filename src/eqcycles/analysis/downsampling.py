import numpy as np
import meshio
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from eqcycles.core.data import SimulationData
from numba import njit, prange
from joblib import Parallel, delayed

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

@njit(parallel=True)
def _aggregate_1d(field, indices, offsets, weights, n_target):
    """Numba-accelerated aggregation for 1D fields (e.g., params)."""
    target_field = np.zeros(n_target, dtype=field.dtype)
    for j in prange(n_target):
        start = offsets[j]
        end = offsets[j+1]
        val = 0.0
        total_w = 0.0
        for k in range(start, end):
            idx = indices[k]
            w = weights[k]
            val += field[idx] * w
            total_w += w
        if total_w > 0:
            target_field[j] = val / total_w
    return target_field

@njit(parallel=True)
def _aggregate_2d(field, indices, offsets, weights, n_target):
    """Numba-accelerated aggregation for 2D fields (time-dependent)."""
    ntime = field.shape[1]
    target_field = np.zeros((n_target, ntime), dtype=field.dtype)
    for j in prange(n_target):
        start = offsets[j]
        end = offsets[j+1]
        
        # Determine total weight for normalization
        total_w = 0.0
        for k in range(start, end):
            total_w += weights[k]
            
        if total_w > 0:
            for k in range(start, end):
                idx = indices[k]
                w = weights[k]
                # Factor out total_w for efficiency
                norm_w = w / total_w
                for t in range(ntime):
                    target_field[j, t] += field[idx, t] * norm_w
                    
    return target_field

def downsample_simulation(
    high_res_data: SimulationData, 
    target_mesh: Union[meshio.Mesh, str, Path],
    scale_factor: float = 1.2,
    z_limit: float = -18.0,
    output_dir: Optional[str] = None,
    run_id: str = "downsampled",
    n_jobs: int = -1
) -> SimulationData:
    """
    Downsamples simulation data from high-resolution to low-resolution with Numba and Joblib.
    
    Args:
        high_res_data: The source SimulationData object.
        target_mesh: The target meshio.Mesh object or path.
        scale_factor: Multiplier for search radius (default 1.2).
        z_limit: Depth threshold for boundary filtering (default -18.0 km).
        output_dir: Optional directory to save the downsampled data.
        run_id: Suffix for saved files if output_dir is provided.
        n_jobs: Number of parallel jobs for field processing (default -1, all CPUs).
        
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
    
    # 3. Mapping Generation
    def find_neighbors(j):
        radius = scale_factor * np.sqrt(target_areas[j] / np.pi)
        indices = tree.query_ball_point(target_centroids[j], radius)
        
        # Filtering
        z_target = target_centroids[j, 2]
        filtered_indices = [
            idx for idx in indices 
            if not ((z_target > z_limit and source_centroids[idx, 2] < z_limit) or 
                    (z_target < z_limit and source_centroids[idx, 2] > z_limit))
        ]
        
        if not filtered_indices:
            _, idx = tree.query(target_centroids[j], k=1)
            filtered_indices = [idx]
        return filtered_indices

    target_neighbor_indices = Parallel(n_jobs=n_jobs)(
        delayed(find_neighbors)(j) for j in range(n_target)
    )
    
    source_claim_counts = np.zeros(n_source)
    for neighbors in target_neighbor_indices:
        for idx in neighbors:
            source_claim_counts[idx] += 1

    # Flatten mapping for Numba
    indices_flat = np.concatenate(target_neighbor_indices).astype(np.int64)
    offsets = np.zeros(n_target + 1, dtype=np.int64)
    for i, neighbors in enumerate(target_neighbor_indices):
        offsets[i+1] = offsets[i] + len(neighbors)
        
    # Pre-calculate weights: source_area / claim_count
    # These will be applied inside JIT functions
    weights_flat = np.zeros(len(indices_flat), dtype=np.float64)
    for j in range(n_target):
        for k in range(offsets[j], offsets[j+1]):
            src_idx = indices_flat[k]
            weights_flat[k] = source_areas[src_idx] / source_claim_counts[src_idx]

    # 4. Data Aggregation with Joblib
    def apply_mapping(field: np.ndarray) -> Optional[np.ndarray]:
        if field is None:
            return None
        if field.ndim == 1:
            return _aggregate_1d(field, indices_flat, offsets, weights_flat, n_target)
        else:
            return _aggregate_2d(field, indices_flat, offsets, weights_flat, n_target)

    # List of fields to process in parallel
    field_names = ['slip', 'shear_stress', 'normal_stress', 'state_variable', 'slip_rate', 'eq_slip']
    fields = [getattr(high_res_data, name) for name in field_names]
    
    results = Parallel(n_jobs=n_jobs)(delayed(apply_mapping)(f) for f in fields)
    field_mapping = dict(zip(field_names, results))

    # 5. Catalog Mapping
    new_catalog = high_res_data.catalog.copy()
    if 'Hypo_Node' in new_catalog.columns:
        target_tree = cKDTree(target_centroids)
        source_nodes = high_res_data.catalog['Hypo_Node'].values.astype(int)
        hypo_coords = source_centroids[source_nodes]
        _, target_nodes = target_tree.query(hypo_coords, k=1)
        new_catalog['Hypo_Node'] = target_nodes

    # 6. Construct new SimulationData
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
        eq_slip=field_mapping['eq_slip'],
        catalog=new_catalog,
        slip_rate=field_mapping['slip_rate'],
        state_variable=field_mapping['state_variable'],
        shear_stress=field_mapping['shear_stress'],
        normal_stress=field_mapping['normal_stress'],
        slip=field_mapping['slip']
    )
    
    if output_dir:
        result.save(output_dir, run_id=run_id)
        
    return result
