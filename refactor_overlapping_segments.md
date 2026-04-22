# Refactor Plan: Support for Overlapping Fault Segments

This document outlines the changes required in the `eqcycles` package to support earthquake cycle simulations with overlapping fault segments (e.g., East, Center, West segments with step-overs).

## Problem Statement
The current implementation projects all simulation mesh nodes onto a single, simplified 1D fault trace. In new simulations where segments overlap (sharing longitude but having different latitudes), this "collapse" merges parallel faults into a single along-strike coordinate, losing the physical separation and potentially corrupting rupture detection and scoring.

## Proposed Architectural Changes

### 1. Core Data & IO
*   **`src/eqcycles/core/data.py`**: Add `node_tags` to `SimulationData`. This will store the Physical Surface name for each mesh node.
*   **`src/eqcycles/io/hbi.py`**: Update `HBILoader` to parse GMSH physical surfaces. It should map the cell-based tags in the `.msh` file to the node-based coordinates used in the simulation.

### 2. Geometry & Projection
*   **`src/eqcycles/analysis/geometry.py`**:
    *   Modify `project_to_fault_trace` to accept a shapefile with multiple line segments.
    *   The function should use `node_tags` to ensure nodes tagged as "Fault_East" are only projected onto the "East" segment of the shapefile.
    *   Determine a strategy for the `along_strike` coordinate:
        *   *Option A (Longitudinal)*: Use longitude-based distance for the X-axis. Overlapping segments will overlap on the plot.
        *   *Option B (Linearized)*: Define segments with offsets (e.g., East: 0-100km, Center: 100-200km) to keep them separate in 1D analysis.
        *   **Recommendation**: Use Option A for OT scoring compatibility but add metadata about segments.

### 3. Rupture & Sequence Analysis
*   **`src/eqcycles/analysis/rupture.py`**: 
    *   Update `export_ruptures_to_geodataframe` to identify which segment geometry to cut for each event. If an event ruptures multiple segments, it should return a `MultiLineString`.
*   **`src/eqcycles/analysis/sequences.py`**:
    *   Update coverage logic. Coverage fraction should be calculated as the union of ruptured intervals across all segments relative to the total "unique" fault length, or as a weighted sum of segment lengths.

### 4. Optimal Transport Scoring & Point Selection
*   **`src/eqcycles/analysis/scoring.py`**:
    *   Ensure `prepare_sim_event_data` uses the updated segment-aware projection. 
    *   Consider if the "mass" (rupture length) should be calculated differently if multiple parallel segments rupture.
*   **`src/eqcycles/analysis/point_selection.py`**:
    *   Update `find_node_at_point` to allow filtering by segment tag. If two segments overlap at the same along-strike distance, the user should be able to specify which segment's node to retrieve.

### 5. Visualization
*   **`src/eqcycles/vis/rupture_sequence_matplotlib.py`**:
    *   Update `plot_rupture_sequence_matplotlib` to distinguish segments.
    *   If using Option A (Longitudinal X-axis), differentiate segments using colors or slight vertical offsets for markers.
*   **`src/eqcycles/vis/diagnostics.py`**:
    *   Update `plot_3d_snapshot` to allow coloring the mesh by Physical Surface tag to verify the tagging process.

### 6. Mesh Processing
*   **`src/eqcycles/analysis/downsampling.py`**:
    *   Update `downsample_simulation` to carry over the `node_tags`. The downsampled node should take the tag of its nearest high-res neighbor or the most frequent tag among its source neighbors.

## Implementation Steps

1.  **Tag Extraction**: Implement tag reading in `HBILoader`.
2.  **Multi-Segment Projection**: Update `geometry.py` to handle the 3-segment shapefile.
3.  **Validation Plot**: Create a diagnostic plot showing the mesh colored by tags and the shapefile segments overlaid.
4.  **Analysis Update**: Fix `rupture.py` and `sequences.py` to handle multi-segment ruptures.
5.  **Refactor Scoring**: Ensure OT scores are recalculated correctly with the new geometry.
