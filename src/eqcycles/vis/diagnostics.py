import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from typing import List, Tuple, Optional
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from eqcycles.core.data import SimulationData
from eqcycles.analysis.point_selection import find_node_at_point


def plot_mesh_field(
    mesh_verts: np.ndarray,
    values: np.ndarray,
    mesh_limits: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "plasma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    label: str = "",
    title: str = "",
    colorbar: bool = True,
    edge_color: str = "none",
    edge_linewidth: float = 0.2,
    alpha: float = 1.0,
    z_exaggeration: float = 5.0,
    elev: float = 25.0,
    azim: float = -70.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Low-level renderer for a scalar field on a 3D triangular mesh.

    Args:
        mesh_verts:      Triangle vertex array of shape (N, 3, 3) — one row per
                         face, each containing the XYZ coordinates of its 3 vertices.
        values:          Scalar field of shape (N,) — one value per face.
        mesh_limits:     Optional [xmin, xmax, ymin, ymax, zmin, zmax] axis bounds.
        ax:              Existing 3D Axes. If None a new figure is created.
        cmap:            Matplotlib colormap name.
        vmin, vmax:      Color-scale bounds; defaults to 5th/95th percentile.
        label:           Colorbar / field label.
        title:           Axes title.
        colorbar:        Whether to attach a colorbar.
        edge_color:      Triangle edge color ('none', 'k', '#333333', …).
        edge_linewidth:  Line width for triangle edges (used when edge_color != 'none').
        alpha:           Face transparency [0, 1].
        z_exaggeration:  Vertical stretch factor applied to the box aspect ratio.
        elev, azim:      Camera elevation and azimuth in degrees.

    Returns:
        (fig, ax) tuple.
    """
    # ── Figure / axes ──────────────────────────────────────────────────────────
    if ax is None:
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # ── Color scaling ──────────────────────────────────────────────────────────
    _vmin = float(np.nanpercentile(values, 5))  if vmin is None else vmin
    _vmax = float(np.nanpercentile(values, 95)) if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)

    face_colors = plt.get_cmap(cmap)(norm(values))
    if alpha < 1.0:
        face_colors[:, 3] = alpha

    # ── Mesh collection ────────────────────────────────────────────────────────
    lw = edge_linewidth if edge_color != "none" else 0
    poly = Poly3DCollection(
        mesh_verts,
        facecolors=face_colors,
        edgecolors=edge_color,
        linewidths=lw,
        antialiased=True,
        shade=False,
    )
    ax.add_collection3d(poly)

    # ── Axis limits & aspect ───────────────────────────────────────────────────
    if mesh_limits is not None:
        ax.set_xlim(mesh_limits[0], mesh_limits[1])
        ax.set_ylim(mesh_limits[2], mesh_limits[3])
        ax.set_zlim(mesh_limits[4], mesh_limits[5])

        dx = abs(mesh_limits[1] - mesh_limits[0]) or 1.0
        dy = abs(mesh_limits[3] - mesh_limits[2]) or 1.0
        dz = abs(mesh_limits[5] - mesh_limits[4]) or 1.0
        ax.set_box_aspect([dx, dy, dz * z_exaggeration])

    ax.view_init(elev=elev, azim=azim)

    # ── Professional styling ───────────────────────────────────────────────────
    # Transparent panes with subtle grid lines
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#cccccc")
        pane.set_linewidth(0.5)

    ax.grid(True, color="#e8e8e8", linewidth=0.4)

    # Tick labels
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(labelsize=10, pad=3)

    # Axis labels (units match the mesh coordinate system — metres by default)
    ax.set_xlabel("X (m)", fontsize=9, labelpad=4)
    ax.set_ylabel("Y (m)", fontsize=9, labelpad=4)
    ax.set_zlabel("Z (m)", fontsize=9, labelpad=4)

    # ── Colorbar ───────────────────────────────────────────────────────────────
    if colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(
            mappable, ax=ax,
            shrink=0.30, pad=0.04, aspect=35,
            orientation="vertical",
        )
        cbar.set_label(label, fontsize=10, labelpad=6)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(0.5)

    # ── Title ──────────────────────────────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=-10)

    return fig, ax


def plot_3d_snapshot(
    sim_data: SimulationData,
    shapefile_path: str,
    param: str,
    target_time_year: float,
    points_to_mark: Optional[List[Tuple[float, float]]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders a 3D view of a scalar parameter field at a chosen time slice.

    Extracts the field snapshot from *sim_data*, delegates rendering to
    :func:`plot_mesh_field`, and optionally marks specific fault-trace points.

    Args:
        sim_data:          Loaded simulation data.
        shapefile_path:    Path to the fault-trace shapefile (used for point lookup).
        param:             Name of the SimulationData attribute to visualise
                           (e.g. 'slip_rate', 'shear_stress').
        target_time_year:  Target time in years; the nearest available snapshot is used.
        points_to_mark:    List of (dist_along_strike_km, depth_km) tuples to highlight
                           with star markers.
        ax:                Existing 3D Axes. If None a new figure is created.
        **kwargs:          Forwarded to :func:`plot_mesh_field`:
                           ``cmap``, ``vmin``, ``vmax``, ``edge_color``,
                           ``edge_linewidth``, ``alpha``, ``z_exaggeration``,
                           ``elev``, ``azim``.

    Returns:
        (fig, ax) tuple.
    """
    # ── Validate & extract field ───────────────────────────────────────────────
    if not hasattr(sim_data, param):
        raise ValueError(
            f"Parameter '{param}' not found in SimulationData. "
            "Check the attribute name and that you loaded the field."
        )
    param_data = getattr(sim_data, param)
    if param_data is None:
        raise ValueError(f"Parameter '{param}' is None — was it loaded?")
    if param_data.ndim != 2:
        raise ValueError(
            f"Parameter '{param}' must be a 2-D array (n_nodes × n_time_steps), "
            f"got shape {param_data.shape}."
        )

    time_idx = int(np.argmin(np.abs(sim_data.time - target_time_year)))
    values    = param_data[:, time_idx]
    time_val  = float(sim_data.time[time_idx])

    # ── Delegate to low-level renderer ────────────────────────────────────────
    mesh_field_kwargs = {
        k: kwargs[k]
        for k in (
            "cmap", "vmin", "vmax", "edge_color", "edge_linewidth",
            "alpha", "z_exaggeration", "elev", "azim",
        )
        if k in kwargs
    }

    fig, ax = plot_mesh_field(
        mesh_verts=sim_data.mesh_verts,
        values=values,
        mesh_limits=sim_data.mesh_limits,
        ax=ax,
        label=param,
        title=f"{param}   |   t = {time_val:.2f} yr",
        **mesh_field_kwargs,
    )

    # ── Optional point markers ─────────────────────────────────────────────────
    if points_to_mark:
        for dist_km, depth_km in points_to_mark:
            node_idx = find_node_at_point(
                sim_data, shapefile_path, dist_km, depth_km
            )
            x, y, z = sim_data.coords[node_idx]
            ax.scatter(
                x, y, z,
                marker="*", s=250,
                color="white", edgecolor="#222222", linewidths=0.8,
                depthshade=False, zorder=10,
            )

    return fig, ax


def plot_point_timeseries(
    sim_data: SimulationData,
    shapefile_path: str,
    params: List[str],
    point: Tuple[float, float],
    time_window: Optional[Tuple[float, float]] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the time series of one or more parameters at a single mesh node,
    using separate vertically-stacked subplots for each parameter.

    Args:
        sim_data:      The loaded simulation data.
        shapefile_path: Path to the fault trace shapefile.
        params:        List of parameter names to plot.
        point:         (distance_along_strike_km, depth_km) of the target point.
        time_window:   Optional (min_t, max_t) x-axis limits.
        fig, axes:     Existing figure/axes to draw on; both must be supplied together.
        **kwargs:      Passed to ``plt.subplots`` (e.g. ``figsize``).

    Returns:
        (fig, axes) tuple.
    """
    if not params:
        raise ValueError("The 'params' list cannot be empty.")

    if fig is None or axes is None:
        fig, axes = plt.subplots(
            len(params), 1,
            figsize=kwargs.get("figsize", (10, 5 * len(params))),
            sharex=True,
        )
    axes = np.atleast_1d(axes)

    target_dist_km, target_depth_km = point
    node_index = find_node_at_point(
        sim_data, shapefile_path, target_dist_km, target_depth_km
    )

    for param, ax in zip(params, axes):
        if not hasattr(sim_data, param):
            print(f"Warning: Parameter '{param}' not found in SimulationData.")
            ax.set_ylabel(f"'{param}' not found")
            continue

        data_series = getattr(sim_data, param)[node_index, :]
        ax.plot(sim_data.time, data_series, label=param)
        ax.set_ylabel(param, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    if len(axes) > 0:
        axes[-1].set_xlabel("Time (years)", fontsize=12)

    fig.suptitle(
        f"Time Series at (dist={target_dist_km} km, depth={target_depth_km} km)",
        fontsize=14, y=0.99,
    )

    if time_window:
        axes[0].set_xlim(time_window)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axes
