import numpy as np
import meshio
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from eqcycles.core.data import SimulationData
from numba import njit, prange

def calculate_cell_areas(mesh: meshio.Mesh) -> np.ndarray:
    areas = []
    points = mesh.points
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            v0 = points[triangles[:, 0]]
            v1 = points[triangles[:, 1]]
            v2 = points[triangles[:, 2]]
            cross_product = np.cross(v1 - v0, v2 - v0)
            block_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
            areas.append(block_areas)
        elif cell_block.type == "quad":
            quads = cell_block.data
            v0 = points[quads[:, 0]]
            v1 = points[quads[:, 1]]
            v2 = points[quads[:, 2]]
            v3 = points[quads[:, 3]]
            # Divide quad into two triangles (0-1-2 and 0-2-3)
            area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            area2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0), axis=1)
            areas.append(area1 + area2)
    
    if not areas:
        raise ValueError("Mesh does not contain 'triangle' or 'quad' cells.")
    
    return np.concatenate(areas)

def get_cell_centroids(mesh: meshio.Mesh) -> np.ndarray:
    centroids = []
    points = mesh.points
    for cell_block in mesh.cells:
        if cell_block.type in ["triangle", "quad"]:
            block_centroids = points[cell_block.data].mean(axis=1)
            centroids.append(block_centroids)
            
    if not centroids:
        raise ValueError("Mesh does not contain 'triangle' or 'quad' cells.")
        
    return np.vstack(centroids)

@njit(parallel=True)
def _aggregate_1d_numba(field, indices, offsets, weights, n_target):
    """Numba-accelerated parallel aggregation for 1D fields."""
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
def _aggregate_2d_numba(field, indices, offsets, weights, n_target):
    """Numba-accelerated parallel aggregation for 2D fields."""
    ntime = field.shape[1]
    target_field = np.zeros((n_target, ntime), dtype=field.dtype)
    for j in prange(n_target):
        start = offsets[j]
        end = offsets[j+1]
        
        total_w = 0.0
        for k in range(start, end):
            total_w += weights[k]
            
        if total_w > 0:
            for k in range(start, end):
                idx = indices[k]
                w = weights[k]
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
    run_id: str = "downsampled"
) -> SimulationData:
    """
    Downsamples simulation data using Numba's internal multithreading.
    """
    # 1. Geometry and Area Extraction
    if isinstance(target_mesh, (str, Path)):
        target_mesh = meshio.read(target_mesh)
    source_areas = calculate_cell_areas(high_res_data.mesh)
    source_centroids = get_cell_centroids(high_res_data.mesh)
    target_areas = calculate_cell_areas(target_mesh)
    target_centroids = get_cell_centroids(target_mesh)
    
    n_source = len(source_centroids)
    n_target = len(target_centroids)
    
    # 2. Vectorized Spatial Indexing
    tree = cKDTree(source_centroids)
    radii = scale_factor * np.sqrt(target_areas / np.pi)
    all_neighbors = tree.query_ball_point(target_centroids, radii)
    
    # 3. Process Mapping and Claim Counts
    target_neighbor_indices = []
    source_claim_counts = np.zeros(n_source)
    source_z = source_centroids[:, 2]
    
    for j, indices in enumerate(all_neighbors):
        z_target = target_centroids[j, 2]
        filtered = [
            idx for idx in indices 
            if not ((z_target > z_limit and source_z[idx] < z_limit) or 
                    (z_target < z_limit and source_z[idx] > z_limit))
        ]
        if not filtered:
            _, idx = tree.query(target_centroids[j], k=1)
            filtered = [idx]
        target_neighbor_indices.append(filtered)
        for idx in filtered:
            source_claim_counts[idx] += 1

    # Flatten mapping for Numba
    indices_flat = np.concatenate(target_neighbor_indices).astype(np.int64)
    offsets = np.zeros(n_target + 1, dtype=np.int64)
    for i, neighbors in enumerate(target_neighbor_indices):
        offsets[i+1] = offsets[i] + len(neighbors)
        
    weights_flat = np.zeros(len(indices_flat), dtype=np.float64)
    for j in range(n_target):
        for k in range(offsets[j], offsets[j+1]):
            src_idx = indices_flat[k]
            weights_flat[k] = source_areas[src_idx] / source_claim_counts[src_idx]

    # 4. Data Aggregation (Sequential calls to Parallel JIT functions)
    def apply_mapping(field: np.ndarray) -> Optional[np.ndarray]:
        if field is None: return None
        if field.ndim == 1:
            return _aggregate_1d_numba(field, indices_flat, offsets, weights_flat, n_target)
        else:
            return _aggregate_2d_numba(field, indices_flat, offsets, weights_flat, n_target)

    field_mapping = {
        'slip': apply_mapping(high_res_data.slip),
        'shear_stress': apply_mapping(high_res_data.shear_stress),
        'normal_stress': apply_mapping(high_res_data.normal_stress),
        'state_variable': apply_mapping(high_res_data.state_variable),
        'slip_rate': apply_mapping(high_res_data.slip_rate),
        'eq_slip': apply_mapping(high_res_data.eq_slip)
    }

    # 5. Catalog Mapping
    new_catalog = high_res_data.catalog.copy()
    if 'Hypo_Node' in new_catalog.columns:
        target_tree = cKDTree(target_centroids)
        source_nodes = high_res_data.catalog['Hypo_Node'].values.astype(int)
        hypo_coords = source_centroids[source_nodes]
        _, target_nodes = target_tree.query(hypo_coords, k=1)
        new_catalog['Hypo_Node'] = target_nodes

    # 6. Construct new SimulationData
    # Collect all face vertices for target mesh plotting
    mesh_verts = []
    for cell_block in target_mesh.cells:
        if cell_block.type in ["triangle", "quad"]:
            for cell_indices in cell_block.data:
                mesh_verts.append(target_mesh.points[cell_indices])
    
    if len(mesh_verts) > 0 and all(len(v) == len(mesh_verts[0]) for v in mesh_verts):
        mesh_verts = np.array(mesh_verts)

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
