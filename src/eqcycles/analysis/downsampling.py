import os
import shutil
import numpy as np
import meshio
from dataclasses import dataclass
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
        end = offsets[j + 1]
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
        end = offsets[j + 1]
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


# ---------------------------------------------------------------------------
# Field name → HBI file prefix mapping
# ---------------------------------------------------------------------------

_FIELD_FILE_MAP = {
    'slip_rate':      'vel',
    'state_variable': 'psi',
    'shear_stress':   'tau',
    'normal_stress':  'sigma',
    'slip':           'slip',
}

_ALL_HEAVY_FIELDS = list(_FIELD_FILE_MAP.keys())


# ---------------------------------------------------------------------------
# SpatialMapping dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpatialMapping:
    """Precomputed area-weighted spatial mapping from source to target mesh."""
    indices_flat: np.ndarray     # (M,) int64 — flattened source neighbour indices
    offsets: np.ndarray          # (n_target+1,) int64 — CSR-style row offsets
    weights_flat: np.ndarray     # (M,) float64 — area-weighted per entry
    n_source: int
    n_target: int
    source_centroids: np.ndarray  # (n_source, 3)
    target_centroids: np.ndarray  # (n_target, 3)
    target_mesh: meshio.Mesh


# ---------------------------------------------------------------------------
# Mapping construction (geometry only, no heavy fields)
# ---------------------------------------------------------------------------

def build_spatial_mapping(
    source: Union[SimulationData, meshio.Mesh],
    target_mesh: Union[meshio.Mesh, str, Path],
    scale_factor: float = 1.2,
    z_limit: float = -18.0,
) -> SpatialMapping:
    """
    Builds the area-weighted spatial mapping from source to target mesh.

    Only mesh geometry is required — heavy fields need not be loaded.

    Args:
        source: Either a SimulationData (uses .mesh) or a meshio.Mesh directly.
        target_mesh: Target meshio.Mesh, or a path to a mesh file.
        scale_factor: Neighbour search radius = scale_factor * sqrt(target_area / pi).
        z_limit: Depth boundary (km) that prevents mixing across the surface.

    Returns:
        A SpatialMapping dataclass ready for use with stream_spatially_downsample
        or downsample_simulation.
    """
    if isinstance(source, SimulationData):
        source_mesh = source.mesh
    else:
        source_mesh = source
    if isinstance(target_mesh, (str, Path)):
        target_mesh = meshio.read(target_mesh)

    source_areas = calculate_cell_areas(source_mesh)
    source_centroids = get_cell_centroids(source_mesh)
    target_areas = calculate_cell_areas(target_mesh)
    target_centroids = get_cell_centroids(target_mesh)

    n_source = len(source_centroids)
    n_target = len(target_centroids)

    tree = cKDTree(source_centroids)
    radii = scale_factor * np.sqrt(target_areas / np.pi)
    all_neighbors = tree.query_ball_point(target_centroids, radii)

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

    indices_flat = np.concatenate(target_neighbor_indices).astype(np.int64)
    offsets = np.zeros(n_target + 1, dtype=np.int64)
    for i, neighbors in enumerate(target_neighbor_indices):
        offsets[i + 1] = offsets[i] + len(neighbors)

    weights_flat = np.zeros(len(indices_flat), dtype=np.float64)
    for j in range(n_target):
        for k in range(offsets[j], offsets[j + 1]):
            src_idx = indices_flat[k]
            weights_flat[k] = source_areas[src_idx] / source_claim_counts[src_idx]

    return SpatialMapping(
        indices_flat=indices_flat,
        offsets=offsets,
        weights_flat=weights_flat,
        n_source=n_source,
        n_target=n_target,
        source_centroids=source_centroids,
        target_centroids=target_centroids,
        target_mesh=target_mesh,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_field_info(source_dir: Union[str, Path], run_id) -> Tuple[int, int]:
    """Returns (ncell, ntime) from xyz and time files — no heavy fields loaded."""
    base = Path(source_dir)
    coords = np.loadtxt(base / f"xyz{run_id}.dat")
    ncell = len(coords)
    time_raw = np.loadtxt(base / f"time{run_id}.dat")
    time_vec = time_raw[:, 1] if time_raw.ndim > 1 else time_raw
    ntime = len(time_vec)
    return ncell, ntime


# ---------------------------------------------------------------------------
# Streaming spatial downsampling
# ---------------------------------------------------------------------------

def stream_spatially_downsample(
    source_dir: Union[str, Path],
    run_id,
    mapping: SpatialMapping,
    output_dir: Union[str, Path],
    out_run_id,
    fields: Optional[List[str]] = None,
    k_temporal: int = 1,
) -> None:
    """
    Spatially downsamples HBI binary fields from source to target mesh,
    streaming one time step at a time.

    Peak RAM is O(2 * max(n_source, n_target)) per field — independent of ntime.

    Args:
        source_dir: Directory containing source HBI binary files.
        run_id: Source run identifier (file suffix).
        mapping: Precomputed SpatialMapping from build_spatial_mapping.
        output_dir: Directory to write output files.
        out_run_id: Output run identifier (file suffix).
        fields: Heavy fields to process. Defaults to all five.
        k_temporal: Temporal stride — keep every k-th time step (1 = all steps).
    """
    if fields is None:
        fields = _ALL_HEAVY_FIELDS

    base = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ncell, ntime = _read_field_info(source_dir, run_id)
    kept_steps = list(range(0, ntime, k_temporal))

    # Metadata: target coordinates
    np.savetxt(out / f"xyz{out_run_id}.dat", mapping.target_centroids)

    # Time: preserve original format (1-col or 2-col), subsample rows
    time_raw = np.loadtxt(base / f"time{run_id}.dat")
    np.savetxt(out / f"time{out_run_id}.dat", time_raw[kept_steps])

    # Catalog: remap Hypo_Node column (index 4) from source to target mesh
    catalog_src = base / f"event{run_id}.dat"
    try:
        catalog = pd.read_csv(catalog_src, sep=r'\s+', header=None)
        if not catalog.empty and catalog.shape[1] >= 5:
            target_tree = cKDTree(mapping.target_centroids)
            source_nodes = catalog.iloc[:, 4].values.astype(int)
            hypo_coords = mapping.source_centroids[source_nodes]
            _, target_nodes = target_tree.query(hypo_coords, k=1)
            catalog.iloc[:, 4] = target_nodes
        catalog.to_csv(out / f"event{out_run_id}.dat", sep=' ', index=False, header=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        shutil.copy(catalog_src, out / f"event{out_run_id}.dat")

    # EQslip: spatially aggregate event-by-event (not a time-series field)
    eq_slip_src = base / f"EQslip{run_id}.dat"
    file_bytes = os.path.getsize(eq_slip_src)
    num_events = file_bytes // (ncell * 8)
    eq_mmap = np.memmap(eq_slip_src, dtype=np.float64, mode='r', shape=(num_events, ncell))
    with open(out / f"EQslip{out_run_id}.dat", 'wb') as f_out:
        for e in range(num_events):
            row = np.array(eq_mmap[e, :], dtype=np.float64)
            target_row = _aggregate_1d_numba(
                row, mapping.indices_flat, mapping.offsets,
                mapping.weights_flat, mapping.n_target,
            )
            target_row.astype(np.float64).tofile(f_out)
    del eq_mmap

    # Heavy fields: spatially aggregate one time step at a time
    for field in fields:
        file_prefix = _FIELD_FILE_MAP[field]
        src_path = base / f"{file_prefix}{run_id}.dat"
        if not src_path.exists():
            continue
        src_mmap = np.memmap(src_path, dtype=np.float64, mode='r', shape=(ntime, ncell))
        out_path = out / f"{file_prefix}{out_run_id}.dat"
        with open(out_path, 'wb') as f_out:
            for t in kept_steps:
                row = np.array(src_mmap[t, :], dtype=np.float64)
                # vel file stores raw velocity; HBILoader applies log10 in memory.
                # Aggregate in log space to match downsample_simulation, then
                # convert back to raw velocity for storage.
                if field == 'slip_rate':
                    row = np.log10(np.abs(row))
                target_row = _aggregate_1d_numba(
                    row, mapping.indices_flat, mapping.offsets,
                    mapping.weights_flat, mapping.n_target,
                )
                if field == 'slip_rate':
                    target_row = 10.0 ** target_row
                target_row.astype(np.float64).tofile(f_out)
        del src_mmap


# ---------------------------------------------------------------------------
# Streaming temporal downsampling
# ---------------------------------------------------------------------------

def stream_temporally_downsample(
    source_dir: Union[str, Path],
    run_id,
    output_dir: Union[str, Path],
    out_run_id,
    fields: Optional[List[str]] = None,
    k: int = 10,
) -> None:
    """
    Temporally downsamples HBI binary fields by keeping every k-th time step,
    streaming one row at a time.

    Peak RAM is O(ncell) per field — independent of ntime.

    Args:
        source_dir: Directory containing source HBI binary files.
        run_id: Source run identifier (file suffix).
        output_dir: Directory to write output files.
        out_run_id: Output run identifier (file suffix).
        fields: Heavy fields to process. Defaults to all five.
        k: Temporal stride — keep every k-th time step.
    """
    if fields is None:
        fields = _ALL_HEAVY_FIELDS

    base = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ncell, ntime = _read_field_info(source_dir, run_id)
    kept_steps = list(range(0, ntime, k))

    # Metadata: spatial info unchanged, time subsampled
    shutil.copy(base / f"xyz{run_id}.dat", out / f"xyz{out_run_id}.dat")
    shutil.copy(base / f"event{run_id}.dat", out / f"event{out_run_id}.dat")
    shutil.copy(base / f"EQslip{run_id}.dat", out / f"EQslip{out_run_id}.dat")

    time_raw = np.loadtxt(base / f"time{run_id}.dat")
    np.savetxt(out / f"time{out_run_id}.dat", time_raw[kept_steps])

    # Stream heavy fields, keeping only the selected time steps
    for field in fields:
        file_prefix = _FIELD_FILE_MAP[field]
        src_path = base / f"{file_prefix}{run_id}.dat"
        if not src_path.exists():
            continue
        src_mmap = np.memmap(src_path, dtype=np.float64, mode='r', shape=(ntime, ncell))
        out_path = out / f"{file_prefix}{out_run_id}.dat"
        with open(out_path, 'wb') as f_out:
            for t in kept_steps:
                src_mmap[t, :].astype(np.float64).tofile(f_out)
        del src_mmap


# ---------------------------------------------------------------------------
# Original in-memory downsampler (unchanged interface, delegates internally)
# ---------------------------------------------------------------------------

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

    All heavy fields must already be loaded in high_res_data.
    For large simulations where loading all fields at once is memory-prohibitive,
    use build_spatial_mapping + stream_spatially_downsample instead.
    """
    mapping = build_spatial_mapping(high_res_data, target_mesh, scale_factor, z_limit)
    target_mesh = mapping.target_mesh

    def apply_mapping(field: np.ndarray) -> Optional[np.ndarray]:
        if field is None:
            return None
        if field.ndim == 1:
            return _aggregate_1d_numba(
                field, mapping.indices_flat, mapping.offsets,
                mapping.weights_flat, mapping.n_target,
            )
        else:
            return _aggregate_2d_numba(
                field, mapping.indices_flat, mapping.offsets,
                mapping.weights_flat, mapping.n_target,
            )

    field_mapping = {
        'slip':           apply_mapping(high_res_data.slip),
        'shear_stress':   apply_mapping(high_res_data.shear_stress),
        'normal_stress':  apply_mapping(high_res_data.normal_stress),
        'state_variable': apply_mapping(high_res_data.state_variable),
        'slip_rate':      apply_mapping(high_res_data.slip_rate),
        'eq_slip':        apply_mapping(high_res_data.eq_slip),
    }

    new_catalog = high_res_data.catalog.copy()
    if 'Hypo_Node' in new_catalog.columns:
        target_tree = cKDTree(mapping.target_centroids)
        source_nodes = high_res_data.catalog['Hypo_Node'].values.astype(int)
        hypo_coords = mapping.source_centroids[source_nodes]
        _, target_nodes = target_tree.query(hypo_coords, k=1)
        new_catalog['Hypo_Node'] = target_nodes

    new_node_tags = None
    if high_res_data.node_tags is not None:
        source_tree = cKDTree(mapping.source_centroids)
        _, nearest_source_indices = source_tree.query(mapping.target_centroids, k=1)
        new_node_tags = high_res_data.node_tags[nearest_source_indices]

    mesh_verts = []
    for cell_block in target_mesh.cells:
        if cell_block.type in ["triangle", "quad"]:
            for cell_indices in cell_block.data:
                mesh_verts.append(target_mesh.points[cell_indices])

    if len(mesh_verts) > 0 and all(len(v) == len(mesh_verts[0]) for v in mesh_verts):
        mesh_verts = np.array(mesh_verts)

    mesh_limits = [
        target_mesh.points[:, 0].min(), target_mesh.points[:, 0].max(),
        target_mesh.points[:, 1].min(), target_mesh.points[:, 1].max(),
        target_mesh.points[:, 2].min(), target_mesh.points[:, 2].max(),
    ]

    result = SimulationData(
        time=high_res_data.time,
        coords=mapping.target_centroids,
        mesh=target_mesh,
        mesh_verts=mesh_verts,
        mesh_limits=mesh_limits,
        node_tags=new_node_tags,
        eq_slip=field_mapping['eq_slip'],
        catalog=new_catalog,
        slip_rate=field_mapping['slip_rate'],
        state_variable=field_mapping['state_variable'],
        shear_stress=field_mapping['shear_stress'],
        normal_stress=field_mapping['normal_stress'],
        slip=field_mapping['slip'],
    )
    if output_dir:
        result.save(output_dir, run_id=run_id)
    return result
