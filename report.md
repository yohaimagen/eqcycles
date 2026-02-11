# Analysis of Rupture Visualization Discrepancy

## 1. Overview

This report details the analysis of why the new refactored visualization script produces an incomplete or "cut" rupture plot compared to the original `plot_eq_sequance.py` script.

The investigation focused on comparing the logic in the following files:
- **Old Script:** `plot_eq_sequance.py`
- **New Modules:** 
    - `src/eqcycles/analysis/geometry.py`
    - `src/eqcycles/analysis/rupture.py`
    - `src/eqcycles/vis/rupture_sequence.py`

The root cause of the issue appears to be a critical unit mismatch introduced during the refactoring of the geometric projection function, with a secondary issue related to how rupture segments are drawn.

## 2. Primary Cause: Unit Mismatch in Geometric Projection

The most significant error was found in `src/eqcycles/analysis/geometry.py`.

### Old Logic (`plot_eq_sequance.py::map_mesh_to_strike`)
1.  A k-d tree (a data structure for nearest-neighbor lookups) was built from reference fault trace points. The coordinates for this tree were scaled to **kilometers**.
    ```python
    # trace_x and trace_y are in meters
    trace_points = np.column_stack((trace_x * 1e-3, trace_y * 1e-3, np.zeros_like(trace_x)))
    tree = cKDTree(trace_points)
    ```
2.  This tree was then queried using the simulation's mesh coordinates (`sim.coords`), which are in **meters**.
    ```python
    # sim_coords are in meters
    distances, indices = tree.query(sim_coords[:, :3], k=1)
    ```
This discrepancy (tree in km, query points in meters) would lead to incorrect nearest-neighbor mapping, but due to the specifics of the geometry, it may have appeared to work by chance. The final returned value, `node_along_strike`, was correctly in **meters**.

### New Logic (`geometry.py::project_to_fault_trace`)
During refactoring, this unit mismatch was inadvertently made worse, and the output units were changed.

1.  **The Same Unit Mismatch Persists:** The k-d tree is still built using reference points scaled to **kilometers**, while the query points (`sim_coords`) are passed in their original unit of **meters**. This is the core of the problem. Because the query points (in meters, e.g., `350000`) are numerically very large compared to the tree points (in km, e.g., `350`), the query will almost always find the closest tree point to be one of the extremes of the fault trace, effectively compressing the entire fault geometry into a much smaller region.
    ```python
    # from src/eqcycles/analysis/geometry.py
    # Tree points are in KM
    trace_points_2d = np.column_stack((trace_x * 1e-3, trace_y * 1e-3, np.zeros_like(trace_x)))
    tree = cKDTree(trace_points_2d)
    
    # Query points are in METERS
    distances, indices = tree.query(sim_coords[:, :3], k=1) 
    ```
2.  **Change in Output Units:** The function now returns the along-strike distance in **kilometers**.
    ```python
    # from src/eqcycles/analysis/geometry.py
    return node_along_strike * 1e-3 
    ```
The old script expected and worked with distances in meters. While the new visualization script was adapted to handle kilometers, this inconsistency makes the code harder to debug and was not part of the original, working implementation.

**Conclusion:** The unit mismatch in the k-d tree query is the primary reason the rupture appears "cut." The fault geometry is not being correctly projected, causing most of the rupture to be mapped to a small, incorrect segment of the along-strike distance profile.

## 3. Secondary Issue: Change in Plotting Method

A secondary difference was found in how the ruptures are drawn in `src/eqcycles/vis/rupture_sequence.py`.

- **Old Method:** The script iterated through all the ruptured bins and plotted a small circle (`style='c0.15c'`) for each one. This correctly visualizes discontinuous ruptures (i.e., separate patches).
    ```python
    # from plot_eq_sequance.py
    fig_rup.plot(x=rup_locs, y=np.ones_like(rup_locs)*eq_time, 
                 style='c0.15c', fill='red', pen='0.1p,red')
    ```
- **New Method:** The script finds the minimum and maximum extent of the ruptured bins and draws a single, continuous thick line (`pen='0.15c,red'`) between them.
    ```python
    # from src/eqcycles/vis/rupture_sequence.py
    fig.plot(x=[np.min(rup_locs_km), np.max(rup_locs_km)], 
             y=[eq_time_yr, eq_time_yr], 
             pen='0.15c,red')
    ```
If a rupture occurs in two distinct, separate patches, this new method will incorrectly draw a solid line that fills the un-ruptured gap between them. This could be misleading but is not the cause of the "cut" fault issue.

## 4. Recommended Fixes

1.  **Correct the Unit Mismatch:** The highest priority is to fix the `project_to_fault_trace` function in `src/eqcycles/analysis/geometry.py`. The k-d tree and the query points **must** be in the same unit. The most robust solution is to perform all calculations in meters and only convert to kilometers at the final plotting stage.
2.  **Revert to Original Plotting Style:** To ensure accurate representation of discontinuous ruptures, the plotting logic in `rupture_sequence.py` should be changed back to plotting individual markers for each ruptured bin rather than a single continuous line.
3.  **Standardize Internal Units:** For consistency, all analysis functions should work with and return distances in a standard unit (e.g., meters), as was the case in the original script. Conversion to kilometers should only happen within visualization functions just before plotting.
