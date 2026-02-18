import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib as mpl

from eqcycles.core.data import SimulationData
from eqcycles.analysis.point_selection import find_node_at_point

def plot_point_timeseries(
    sim_data: SimulationData,
    shapefile_path: str,
    params: List[str],
    point: Tuple[float, float],
    time_window: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the time series of one or more parameters at a single point,
    using separate, vertically-stacked subplots for each parameter.

    Args:
        sim_data (SimulationData): The loaded simulation data.
        shapefile_path (str): Path to the fault trace shapefile.
        params (List[str]): List of parameter names to plot.
        point (Tuple[float, float]): The (distance_along_strike_km, depth_km) of the target point.
        time_window (Optional[Tuple[float, float]]): A tuple (min_t, max_t) to set the x-axis limits.
        **kwargs: Additional keyword arguments passed to `plt.subplots` (e.g., `figsize`).

    Returns:
        A tuple containing the matplotlib Figure and an array of the Axes objects.
    """
    if not params:
        raise ValueError("The 'params' list cannot be empty.")

    # Create N subplots, where N is the number of parameters, with a shared x-axis.
    fig, axes = plt.subplots(
        len(params), 1, 
        figsize=kwargs.get('figsize', (12, 4 * len(params))), 
        sharex=True
    )
    # Ensure `axes` is always an array, even if there's only one subplot.
    axes = np.atleast_1d(axes)

    target_dist_km, target_depth_km = point
    
    # 1. Find the node index for the given point.
    node_index = find_node_at_point(sim_data, shapefile_path, target_dist_km, target_depth_km)
    
    # 2. Extract and plot data for each parameter on its own subplot.
    for i, (param, ax) in enumerate(zip(params, axes)):
        if not hasattr(sim_data, param):
            print(f"Warning: Parameter '{param}' not found in SimulationData.")
            ax.set_ylabel(f"'{param}' not found")
            continue
            
        data_series = getattr(sim_data, param)[node_index, :]
        
        # Configure axis properties
        ax.plot(sim_data.time, data_series, label=param)
        ax.set_ylabel(param, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    # 3. Finalize plot.
    # Set x-label only on the bottom subplot
    if len(axes) > 0:
        axes[-1].set_xlabel("Time (years)", fontsize=12)
    
    # Set a main title for the entire figure
    fig.suptitle(f"Time Series at (dist={target_dist_km} km, depth={target_depth_km} km)", fontsize=14, y=0.99)
    
    if time_window:
        axes[0].set_xlim(time_window)

    # Adjust layout to prevent title overlap
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig, axes

def plot_3d_snapshot(
    sim_data: SimulationData,
    shapefile_path: str,
    param: str,
    target_time_year: float,
    points_to_mark: Optional[List[Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a 3D view of a parameter at a single time step.

    Args:
        sim_data (SimulationData): The loaded simulation data.
        shapefile_path (str): Path to the fault trace shapefile.
        param (str): The name of the parameter to plot.
        time_step_index (int): The index of the time step to plot.
        points_to_mark (Optional[List[Tuple[float, float]]]): List of (dist_km, depth_km) points to mark.
        ax (Optional[plt.Axes]): A 3D matplotlib axes object. If None, a new figure is created.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects.
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # 1. Data Selection
    if not hasattr(sim_data, param):
        raise ValueError(f"Parameter '{param}' not found in SimulationData object.")
    
    param_data = getattr(sim_data, param)
    time_step_index = np.argmin(np.abs(sim_data.time - target_time_year))
    param_at_time_step = param_data[:, time_step_index]
    time_val = sim_data.time[time_step_index]

    # 2. Setup 3D Plot
    vmin, vmax = np.percentile(param_at_time_step, [5, 95])  # Robust min/max for color scaling
    vmin = kwargs.get('vmin', vmin)
    vmax = kwargs.get('vmax', vmax)
    cmap = kwargs.get('cmap', 'viridis')
    
    # Data is per-node, but color is per-face. Average node values for each face.
    triangles = sim_data.mesh.cells_dict["triangle"]
     

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    face_colors = plt.get_cmap(cmap)(norm(param_at_time_step))

    poly = Poly3DCollection(sim_data.mesh_verts, facecolors=face_colors, edgecolors='none')
    ax.add_collection3d(poly)

    sim_limits = sim_data.mesh_limits
    ax.set_xlim(sim_limits[0], sim_limits[1])
    ax.set_ylim(sim_limits[2], sim_limits[3])
    ax.set_zlim(sim_limits[4], sim_limits[5])

    ax.set_box_aspect([1, 1, 0.15])
    ax.view_init(elev=15, azim=-90)

    # 3. Finalize Plot Details
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
    cbar.set_label(param)
    ax.set_title(f'{param} at t = {time_val:.2f} yrs')

    # 4. Find and Mark Points
    if points_to_mark:
        for point in points_to_mark:
            target_dist_km, target_depth_km = point
            node_index = find_node_at_point(sim_data, shapefile_path, target_dist_km, target_depth_km)
            
            node_coords_3d = sim_data.coords[node_index]
            
            ax.scatter(
                node_coords_3d[0], node_coords_3d[1], node_coords_3d[2],
                marker='*', color='yellow', edgecolor='black', s=200, depthshade=False, zorder=10
            )

    return fig, ax
