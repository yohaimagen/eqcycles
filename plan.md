# Plan for Diagnostic Plotting Tool

This document outlines the implementation plan for a new diagnostic plotting tool for the `eqcycles` package. The tool will provide two key functionalities:
1.  A 3D snapshot of a physical parameter on the fault at a specific time, with highlighted points.
2.  A time series plot of one or more parameters at a specific point on the fault.

## 1. New Modules and File Structure

To maintain separation of concerns, the new functionality will be organized into two new files: one for the new analysis logic (point selection) and one for the new visualization functions.

```
/
├── src/
│   └── eqcycles/
│       ├── ...
│       ├── analysis/
│       │   ├── ...
│       │   └── point_selection.py  #<-- New analysis helper
│       └── vis/
│           ├── ...
│           └── diagnostics.py        #<-- New plotting module
└── ...
```

*   **`src/eqcycles/analysis/point_selection.py`**: Will contain the logic to find the closest mesh node to a user-specified coordinate.
*   **`src/eqcycles/vis/diagnostics.py`**: Will contain the high-level plotting functions for the user to call.

## 2. Component 1: Point Selection Helper

The user will specify points using physical coordinates `(distance_along_strike, depth)`. We need a robust way to map these to the unstructured mesh nodes.

### `analysis/point_selection.py`

A new function `find_node_at_point` will be created here.

**Function Signature:**
```python
def find_node_at_point(
    sim_data: SimulationData,
    shapefile_path: str,
    target_dist_km: float,
    target_depth_km: float
) -> int:
    """Finds the index of the mesh node closest to a target coordinate."""
```

**Logic:**
1.  **Get Node Coordinates:**
    *   Use `project_to_fault_trace` from `analysis.geometry` to get the along-strike distance (in meters) for every node in `sim_data.coords`.
    *   Get the depth for every node from `sim_data.coords[:, 2]` (assuming Z is depth).
2.  **Create Coordinate Array:** Combine the along-strike distances and depths into an `(N, 2)` numpy array of `(distance_m, depth_m)`.
3.  **Find Closest Node:**
    *   Create a `scipy.spatial.cKDTree` from the coordinate array created in the previous step.
    *   Query the tree with the user's `(target_dist_km * 1000, target_depth_km * 1000)` to find the index of the nearest node.
4.  **Return Index:** Return the integer index of the closest node.

## 3. Component 2: Visualization Module

The new plotting functions will reside in `vis/diagnostics.py`.

### `vis/diagnostics.py`

This file will contain two main functions.

#### A. 3D Snapshot Plot

**Function Signature:**
```python
def plot_3d_snapshot(
    sim_data: SimulationData,
    shapefile_path: str,
    param: str,
    time_step_index: int,
    points_to_mark: Optional[List[Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a 3D view of a parameter at a single time step."""
```

**Logic:**
1.  **Data Selection:** Get the parameter data array from `sim_data` using `getattr(sim_data, param)`. Select the data for the given `time_step_index`.
2.  **Setup 3D Plot:** If no `ax` is provided, create a new 3D subplot. Reuse the `Poly3DCollection` logic from `slip_rate_video.py` to create the colored fault surface.
3.  **Find and Mark Points:**
    *   If `points_to_mark` is provided, loop through each `(dist, depth)` tuple.
    *   Call `find_node_at_point` to get the index for each target point.
    *   Retrieve the 3D coordinates `(x, y, z)` of that node from `sim_data.coords[node_index]`.
    *   Plot a marker on the 3D axes at this location using `ax.scatter`, for instance with a large, visible star marker.
4.  **Finalize Plot:** Add a colorbar, title indicating the parameter and time, and set view angles. Return the figure and axes.

#### B. Point Time Series Plot

**Function Signature:**
```python
def plot_point_timeseries(
    sim_data: SimulationData,
    shapefile_path: str,
    params: List[str],
    point: Tuple[float, float],
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the time series of one or more parameters at a single point."""
```

**Logic:**
1.  **Find Node:** Call `find_node_at_point` with the user's `point` to get the single `node_index`.
2.  **Setup 2D Plot:** If no `ax` is provided, create a new 2D subplot. Set the x-axis label to "Time (years)".
3.  **Extract and Plot Data:**
    *   Loop through the list of parameter names in `params`.
    *   For each `param`, extract the full time series for the `node_index`: `data_series = getattr(sim_data, param)[node_index, :]`.
    *   Plot `sim_data.time` vs. `data_series` on the axes. Assign a label to each line corresponding to the parameter name.
4.  **Finalize Plot:** Add a legend, title indicating the location, and grid. Return the figure and axes.

## 4. Development Workflow

1.  **Implement `point_selection.py`**: Create the `find_node_at_point` function. This is the core dependency for both plotting functions. Add unit tests to verify that it correctly finds known points.
2.  **Implement `plot_point_timeseries`**: This is the simpler of the two plotting functions and will serve as a good test for the point selection helper.
3.  **Implement `plot_3d_snapshot`**: Adapt the 3D plotting code from `slip_rate_video.py` and integrate the point marking logic.
4.  **(Optional)** Create a wrapper function `plot_diagnostics(..., point, time_step, ...)` that uses `plt.subplots` to create a figure with both plots arranged side-by-side for a comprehensive view.
