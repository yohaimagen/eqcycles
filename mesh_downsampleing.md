# Specification: eqcycles Mesh and Simulation Data Downsampling

## Objective
Implement a specialized module within the `eqcycles` package to downsample earthquake sequence simulation data. This module will transform a high-resolution `SimulationData` object into a lower-resolution version based on a target `meshio` mesh, strictly preserving the physics of Rate-and-State friction mechanics and maintaining global conservation of seismic moment.

## Integration Path
* **Module Location:** `src/eqcycles/analysis/downsampling.py`
* **Primary Entry Point:** `downsample_simulation(high_res_data: SimulationData, target_mesh: meshio.Mesh, scale_factor: float = 1.2) -> SimulationData`

## Dependencies
* `numpy`
* `scipy.spatial.cKDTree`
* `meshio`
* `eqcycles.core.data.SimulationData`

## Algorithm Implementation Steps

### 1. Geometry and Area Extraction
* **Source Mesh:** Use `high_res_data.mesh`.
* **Target Mesh:** Use the provided `target_mesh`.
* **Centroids:** Extract 3D centroids for all "triangle" cells in both meshes.
* **Area Calculation:** For each triangle, calculate the surface area using the cross product of its edge vectors: `Area = 0.5 * |(v1 - v0) x (v2 - v0)|`.

### 2. Spatial Indexing
* Build a `scipy.spatial.cKDTree` using the centroids of the **source** (high-resolution) mesh.

### 3. Mapping Matrix Generation
To handle large time-dependent fields efficiently, pre-calculate a sparse mapping.
* **Dynamic Radius Search:** For each cell $j$ in the target mesh:
    * Compute `radius_j = scale_factor * sqrt(Area_target_j / pi)`.
    * Query the tree for source indices $i$ within `radius_j`.
* **Over-sampling Normalization:** Track how many times each source cell $i$ is included in any target cell's radius. Normalize the weight of source cell $i$ by this "claim count" to ensure that the sum of weighted areas in the target domain exactly matches the source domain area.
* **Boundary and Depth Filtering:** Filter indices to prevent smearing across major physical/frictional transitions (e.g., Z-depth boundaries at -18 km). If filtering results in an empty list, fallback to a nearest-neighbor (`K=1`) query.

### 4. Physics-Aware Data Aggregation
The mapping will be applied to the fields within `SimulationData`:

* **Linear Variables (`slip`, `shear_stress`, `normal_stress`, `eq_slip`):** 
  Calculate the area-weighted arithmetic mean to conserve total moment/force.
  `Value_target = Sum(Weight_i * Area_i * Value_i) / Sum(Weight_i * Area_i)`
  
*  **Note on `eq_slip`:** Unlike other time-dependent fields which have a shape of `(ncell, ntime)`, `eq_slip` has a shape of `(ncell, num_events)`. The downsampling mapping must be applied along the spatial axis for both cases. 
  
* **In `eqcycles`, `SimulationData.slip_rate` is stored as $\log_{10}(V)$. To preserve the logarithmic mean of velocity, apply the area-weighted arithmetic mean directly to the logged values.

* **State Variable (`state_variable`):** 
  Apply area-weighted arithmetic mean. While $\psi$ is part of a non-linear system, it is mapped linearly to preserve its spatial distribution during the downsampling.

* **Friction Parameters (a, b, Dc):** 
  If provided, use **nearest-neighbor (K=1)** mapping. Do not average these parameters, as it would blur the sharp frictional transitions that define the physics of the rupture.

### 5. Catalog Mapping
* Update the `catalog` (Pandas DataFrame) in the new `SimulationData` object.
* Map the `Hypo_Node` (hypocenter index) from the source mesh to the closest centroid index in the target mesh.

## Expected Output
A new `SimulationData` object where:
1. `coords` and `mesh` match the low-resolution target.
2. All time-dependent arrays have the same number of timesteps as the input but a reduced spatial dimension (`ncell`).
3. The global seismic moment and fault area are conserved.
