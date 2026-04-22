"""
PyVista-based 3-D diagnostics for fault mesh simulation data.

Mirrors the API of diagnostics.py but uses PyVista/VTK for rendering,
giving proper depth-sorting, real lighting, and interactive Jupyter widgets.

Jupyter setup (run once before plotting)::

    import pyvista as pv
    pv.set_jupyter_backend("trame")   # interactive  (needs trame installed)
    # -- or --
    pv.set_jupyter_backend("static")  # inline PNG, no extra deps
"""

import numpy as np
import pyvista as pv
from typing import List, Tuple, Optional, Union

from eqcycles.core.data import SimulationData
from eqcycles.analysis.point_selection import find_node_at_point


# ── Internal helper ────────────────────────────────────────────────────────────

def _build_polydata(mesh_verts: Union[np.ndarray, list]) -> pv.PolyData:
    """
    Convert a face-vertex array to a PyVista PolyData surface.

    Args:
        mesh_verts:  If np.ndarray: Shape (N, V, 3) — one row per face, each
                     containing the XYZ coordinates of its V vertices.
                     If list: List of arrays, each of shape (V_i, 3).

    Returns:
        pv.PolyData with N cells and unmodified coordinates.
    """
    if isinstance(mesh_verts, np.ndarray) and mesh_verts.ndim == 3:
        n_faces = len(mesh_verts)
        n_verts_per_face = mesh_verts.shape[1]
        points  = mesh_verts.reshape(-1, 3)                       # (N*V, 3)
        idx     = np.arange(n_faces * n_verts_per_face).reshape(n_faces, n_verts_per_face)
        faces   = np.hstack([np.full((n_faces, 1), n_verts_per_face, dtype=np.int64), idx]).ravel()
        return pv.PolyData(points.copy(), faces)
    else:
        # Mixed elements or list of arrays
        points_list = []
        faces_list = []
        current_idx = 0
        for face in mesh_verts:
            n_v = len(face)
            points_list.append(face)
            faces_list.append(n_v)
            faces_list.extend(range(current_idx, current_idx + n_v))
            current_idx += n_v
        
        points = np.vstack(points_list)
        faces = np.array(faces_list, dtype=np.int64)
        return pv.PolyData(points, faces)


# ── Low-level renderer ─────────────────────────────────────────────────────────

def plot_mesh_field(
    mesh_verts: np.ndarray,
    values: np.ndarray,
    cmap: str = "plasma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    label: str = "",
    title: str = "",
    show_edges: bool = False,
    edge_color: str = "#444444",
    edge_width: float = 0.3,
    z_exaggeration: float = 5.0,
    background: str = "white",
    lighting: bool = True,
    window_size: Tuple[int, int] = (1400, 600),
    show: bool = True,
    screenshot: Optional[str] = None,
    plotter: Optional[pv.Plotter] = None,
) -> pv.Plotter:
    """
    Low-level renderer for a scalar field on a 3D triangular mesh.

    Args:
        mesh_verts:     Triangle vertex array of shape (N, 3, 3).
        values:         Scalar field of shape (N,) — one value per face.
        cmap:           Matplotlib-compatible colormap name.
        vmin, vmax:     Color-scale bounds; defaults to 5th / 95th percentile.
        label:          Scalar-bar title.
        title:          Figure title rendered in the viewport.
        show_edges:     Overlay triangle edges on the mesh.
        edge_color:     Edge colour (any CSS / hex string).
        edge_width:     Edge line width in pixels.
        z_exaggeration: Vertical stretch applied before rendering.
        background:     Viewport background colour.
        lighting:       Enable PyVista's headlight lighting model.
        window_size:    (width, height) of the render window in pixels.
        show:           Call ``plotter.show()`` before returning.
        screenshot:     If a file path is given, save a PNG screenshot there
                        (works even when ``show=False``).
        plotter:        Existing Plotter to draw into. If None, a new one is
                        created with the given ``window_size``.

    Returns:
        The PyVista Plotter object (call ``.show()`` yourself if ``show=False``).
    """
    # ── Build mesh ─────────────────────────────────────────────────────────────
    # Coordinates are kept unscaled so that axis labels show real values.
    # Z exaggeration is applied at the renderer level instead.
    mesh = _build_polydata(mesh_verts)
    mesh.cell_data[label or "value"] = values

    # ── Color scale ────────────────────────────────────────────────────────────
    _vmin = float(np.nanpercentile(values, 5))  if vmin is None else vmin
    _vmax = float(np.nanpercentile(values, 95)) if vmax is None else vmax

    # ── Scalar-bar styling ─────────────────────────────────────────────────────
    scalar_bar_args = dict(
        title=label,
        n_labels=5,
        italic=False,
        bold=False,
        title_font_size=14,
        label_font_size=11,
        color="black",
        vertical=True,
        width=0.055,
        height=0.35,
        position_x=0.91,
        position_y=0.32,
        fmt="%.3g",
    )

    # ── Plotter ────────────────────────────────────────────────────────────────
    if plotter is None:
        plotter = pv.Plotter(window_size=list(window_size))

    plotter.set_background(background)

    plotter.add_mesh(
        mesh,
        scalars=label or "value",
        preference="cell",
        cmap=cmap,
        clim=[_vmin, _vmax],
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=edge_width,
        lighting=lighting,
        smooth_shading=True,
        scalar_bar_args=scalar_bar_args,
    )

    # ── Grid + axis labels ─────────────────────────────────────────────────────
    plotter.show_bounds(
        grid="back",
        location="outer",
        ticks="outside",
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=3,
        font_size=9,
        color="gray",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
        bold=False,
    )

    # ── Z exaggeration via plotter scale (labels stay in real coordinates) ─────
    if z_exaggeration != 1.0:
        plotter.set_scale(zscale=z_exaggeration, reset_camera=False)

    # ── Camera — oblique view suited for a horizontal fault ────────────────────
    plotter.camera_position = "xz"
    plotter.camera.elevation = 20
    plotter.reset_camera()
    plotter.camera.zoom(1.8)

    # ── Title — placed just above the mesh, not pinned to window top ──────────
    if title:
        plotter.add_text(
            title,
            position=(0.4, 0.7),   # normalized viewport coords (0=bottom, 1=top)
            font_size=12,
            color="black",
            viewport=True,
        )

    # ── Output ─────────────────────────────────────────────────────────────────
    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=not show)
    elif show:
        plotter.show()

    return plotter


# ── High-level snapshot ────────────────────────────────────────────────────────

def plot_3d_snapshot(
    sim_data: SimulationData,
    shapefile_path: str,
    param: str,
    target_time_year: float,
    points_to_mark: Optional[List[Tuple[float, float]]] = None,
    plotter: Optional[pv.Plotter] = None,
    show: bool = True,
    screenshot: Optional[str] = None,
    **kwargs,
) -> pv.Plotter:
    """
    Render a 3D snapshot of a scalar parameter field at a chosen time slice.

    Extracts the field from *sim_data*, delegates rendering to
    :func:`plot_mesh_field`, and optionally marks fault-trace points with
    star glyphs.

    Args:
        sim_data:          Loaded simulation data.
        shapefile_path:    Path to the fault-trace shapefile (for point lookup).
        param:             SimulationData attribute to visualise
                           (e.g. ``'slip_rate'``, ``'shear_stress'``).
        target_time_year:  Target time in years; nearest snapshot is used.
        points_to_mark:    List of ``(dist_along_strike_km, depth_km)`` tuples
                           to highlight with star markers.
        plotter:           Existing Plotter to draw into.
        show:              Call ``plotter.show()`` before returning.
        screenshot:        Path for a PNG screenshot (optional).
        **kwargs:          Forwarded to :func:`plot_mesh_field`:
                           ``cmap``, ``vmin``, ``vmax``, ``show_edges``,
                           ``edge_color``, ``edge_width``, ``z_exaggeration``,
                           ``background``, ``lighting``, ``window_size``.

    Returns:
        The PyVista Plotter object.
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
            f"Parameter '{param}' must be 2-D (n_nodes × n_time_steps), "
            f"got shape {param_data.shape}."
        )

    time_idx = int(np.argmin(np.abs(sim_data.time - target_time_year)))
    values   = param_data[:, time_idx]
    time_val = float(sim_data.time[time_idx])

    # ── Forward to low-level renderer ─────────────────────────────────────────
    mesh_field_kwargs = {
        k: kwargs[k]
        for k in (
            "cmap", "vmin", "vmax", "show_edges", "edge_color", "edge_width",
            "z_exaggeration", "background", "lighting", "window_size",
        )
        if k in kwargs
    }

    pl = plot_mesh_field(
        mesh_verts=sim_data.mesh_verts,
        values=values,
        label=param,
        title=f"{param}   |   t = {time_val:.2f} yr",
        plotter=plotter,
        show=False,        # defer show until after point markers are added
        screenshot=None,
        **mesh_field_kwargs,
    )

    # ── Optional point markers ─────────────────────────────────────────────────
    if points_to_mark:
        for dist_km, depth_km in points_to_mark:
            node_idx     = find_node_at_point(sim_data, shapefile_path, dist_km, depth_km)
            x, y, z      = sim_data.coords[node_idx]
            marker_cloud = pv.PolyData(np.array([[x, y, z]]))
            pl.add_mesh(
                marker_cloud,
                style="points",
                color="white",
                point_size=18,
                render_points_as_spheres=True,
            )
            # black outline ring for contrast
            pl.add_mesh(
                marker_cloud,
                style="points",
                color="#111111",
                point_size=24,
                render_points_as_spheres=True,
                opacity=0.6,
            )

    # ── Final show / screenshot ────────────────────────────────────────────────
    if screenshot:
        pl.show(screenshot=screenshot, auto_close=not show)
    elif show:
        pl.show()

    return pl
