import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.collections import LineCollection
from typing import Dict, Any, List, Optional, Tuple

from eqcycles.core.data import SimulationData
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.analysis.rupture import get_rupture_mask, analyze_rupture_direction
from eqcycles.analysis.sequences import RuptureSequence

# Default configuration dictionary for plotting
DEFAULT_CONFIG = {
    "rupture_threshold": 0.05,
    "rupture_bin_count": 500,
    "panel_width": 8,
    "panel_height": 15,
    "font_size_title": "12",
    "font_weight_title": "bold",
}

_SEQUENCE_COLORS = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
    "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def plot_rupture_sequence_matplotlib(
    sim_data: SimulationData,
    shapefile_path: str,
    output_path: str = None,
    config: Dict[str, Any] = None,
    ax: plt.Axes = None,
    add_rupture_direction: bool = False,
    ploting_mag_treshold: float = 7.0,
    label_mag_threshold: float = 7.5,
    add_catalog_index: bool = False,
    title: str = "Rupture Sequence",
    rupture_color: str = 'red',
    verbose: bool = False,
    hist_cords: Optional[np.ndarray] = None,
    hist_mass: Optional[np.ndarray] = None,
    best_time: Optional[float] = None,
    hist_color: str = 'black',
    hist_linewidth: float = 2.0,
    hist_alpha: float = 0.9,
    sequences: Optional[List[RuptureSequence]] = None,
    sequence_colors: Optional[List[str]] = None,
    sequence_alpha: float = 0.15,
    label_sequences: bool = True,
    color_sequence_events: bool = True,
    time_window: Optional[tuple] = None,
):
    """
    Generates a rupture sequence plot for a given simulation run using Matplotlib.

    Args:
        sim_data (SimulationData): The loaded and standardized simulation data.
        shapefile_path (str): Path to the reference fault trace shapefile.
        output_path (str): The path to save the output PNG image.
        config (Dict[str, Any], optional): A dictionary to override default plotting settings.
        ax (plt.Axes, optional): A matplotlib axes object to plot on. If None, a new figure and axes are created.
        add_rupture_direction (bool): If True, analyze and plot rupture direction arrows.
        ploting_mag_treshold (float): The minimum magnitude to plot.
        hist_cords (np.ndarray, optional): Historical event coordinates, shape (N, 2):
            [east_position_m, year]. Requires ``best_time`` to be set.
        hist_mass (np.ndarray, optional): Historical rupture lengths in metres, shape (N,).
        best_time (float, optional): Simulation time (years) that corresponds to the
            start of the historical catalog.  Historical events are shifted so that
            ``min(hist_cords[:, 1])`` maps to this simulation time.
        hist_color (str): Line colour for historical ruptures. Default 'black'.
        hist_linewidth (float): Line width for historical ruptures.
        hist_alpha (float): Alpha for historical rupture lines.
        sequences (list of RuptureSequence, optional): Detected rupture sequences to
            overlay on the plot, as returned by ``find_rupture_sequences``.
        sequence_colors (list of str, optional): Cycle of hex/named colours for the
            sequence bands.  Defaults to a built-in Tableau-like palette.
        sequence_alpha (float): Transparency of the horizontal sequence band.
        label_sequences (bool): If True, annotate each band with its sequence ID,
            event count and fault coverage fraction.
        color_sequence_events (bool): If True, events that belong to a sequence are
            plotted in their sequence colour instead of ``rupture_color``.
        time_window (tuple, optional): ``(min_t, max_t)`` in simulation years.
            Only events, sequences and historical entries within this range are
            shown.  Defaults to ``(0, sim_data.time[-1])``.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Build a mapping from catalog event index → (sequence_id, color) for fast lookup
    _seq_colors = sequence_colors or _SEQUENCE_COLORS
    _event_to_seq: dict = {}  # catalog_idx -> (seq_id, color)
    if sequences:
        for seq_id, seq in enumerate(sequences):
            color = _seq_colors[seq_id % len(_seq_colors)]
            for ev_idx in seq.event_indices:
                _event_to_seq[ev_idx] = (seq_id, color)

    print(f"--> Processing Run for Rupture Sequence Plot (Matplotlib)...")
    
    # 1. Perform Geometric Analysis
    mesh_along_strike = project_to_fault_trace(sim_data.coords, shapefile_path)

    # 2. Setup Matplotlib Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(cfg['panel_width'], cfg['panel_height']))
    else:
        fig = ax.get_figure()

    # Define plot region based on analysis results, converting meters to negative km
    max_time = sim_data.time[-1]
    t_start, t_end = time_window if time_window is not None else (0.0, float(max_time))
    t_span = t_end - t_start

    x_min_km = -np.max(mesh_along_strike) * 1e-3
    x_max_km = 0.0

    ax.set_xlim(x_min_km, x_max_km)
    ax.set_ylim(t_start, t_end + t_span * 0.05)  # 5% padding at top
    
    ax.set_xlabel("Distance Along Strike [km]")
    ax.set_ylabel("Time [years]")
    ax.set_title(
        title, 
        fontsize=int(cfg["font_size_title"]), 
        fontweight=cfg["font_weight_title"]
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if sim_data.catalog.empty:
        print("    Warning: No events in the catalog to plot.")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax

    # 3. Iterate Through Events and Plot
    for idx, event in sim_data.catalog.iterrows():
        if event.Mw < ploting_mag_treshold:
            continue
        if not (t_start <= event['Time_year'] <= t_end):
            continue
        
        centers, is_rup = get_rupture_mask(
            sim_data, 
            idx, 
            mesh_along_strike, 
            num_bins=cfg["rupture_bin_count"], 
            slip_threshold=cfg["rupture_threshold"]
        )
        
        if not np.any(is_rup):
            continue
            
        rup_locs_km = -centers[is_rup] * 1e-3
        eq_time_yr = event['Time_year']

        if color_sequence_events and idx in _event_to_seq:
            ev_color = _event_to_seq[idx][1]
        else:
            ev_color = rupture_color

        ax.plot(
            rup_locs_km,
            np.full_like(rup_locs_km, eq_time_yr),
            'o',
            markersize=2,
            color=ev_color,
            markeredgecolor=ev_color,
            markeredgewidth=0.1
        )
        
        if event['Mw'] > label_mag_threshold:
            ax.text(
                np.mean(rup_locs_km), 
                eq_time_yr + (max_time * 0.025),
                f"$M_w{event['Mw']:.1f}$", 
                fontsize=8, 
                fontweight="bold", 
                ha="center",
                va="center"
            )
            
            if add_catalog_index:
                ax.text(
                    np.mean(rup_locs_km) + 0.03 * (x_max_km - x_min_km), # Offset to the right
                    eq_time_yr + (max_time * 0.015), # Slightly below magnitude label
                    f"({idx})",
                    fontsize=7,
                    ha="left", # Align to the left of the offset
                    va="center",
                    color="gray"
                )
            
            if add_rupture_direction:
                # try:
                metrics = analyze_rupture_direction(sim_data, idx, mesh_along_strike, verbose=verbose)
                
                if metrics and metrics.code != 0:
                    arrow_length_data = 30
                    # print(f'arrow_length_data : {arrow_length_data}')
                    arrow_props = dict(
                        head_width=50,
                        head_length=25,
                        fc='k',
                        ec='k',
                        lw=1
                    )
                    x_hypo = -metrics.hypocenter_dist * 1e-3
                    y_hypo = metrics.hypocenter_time / (365 * 24 * 60 * 60)
                    # Plot hypocenter star if data is available
                    if metrics.hypocenter_dist is not None and metrics.hypocenter_time is not None:
                        
                        ax.plot(x_hypo, y_hypo, '*', color='yellow', markersize=10, markeredgecolor='black', zorder=10)

                    # Unilateral, left-pointing arrow (prop. along increasing mesh_along_strike)
                    if metrics.code == 1:
                        dx = -arrow_length_data    # Negative dx = points LEFT
                        ax.arrow(x_hypo, y_hypo, dx, 0, **arrow_props, zorder=10)
                    
                    # Unilateral, right-pointing arrow (prop. along decreasing mesh_along_strike)
                    elif metrics.code == -1:
                        dx = arrow_length_data     # Positive dx = points RIGHT
                        ax.arrow(x_hypo, y_hypo, dx, 0, **arrow_props, zorder=10)

                    # Bilateral, two arrows from hypocenter
                    elif metrics.code == 2 and metrics.hypocenter_dist is not None:

                        # Arrow 1: left-pointing (for propagation along increasing mesh_along_strike)
                        ax.arrow(x_hypo, y_hypo, -arrow_length_data, 0, **arrow_props, zorder=10)
                        
                        # Arrow 2: right-pointing (for propagation along decreasing mesh_along_strike)
                        ax.arrow(x_hypo, y_hypo, arrow_length_data, 0, **arrow_props, zorder=10)

                # except Exception as e:
                #     print(f"    Warning: Direction analysis failed for event {idx}: {e}")

    # --- Overlay rupture sequences ---
    if sequences:
        legend_handles = []
        for seq_id, seq in enumerate(sequences):
            # Skip sequences entirely outside the time window
            if seq.end_time_year < t_start or seq.start_time_year > t_end:
                continue

            color = _seq_colors[seq_id % len(_seq_colors)]

            # Horizontal band spanning the sequence's time range (clamped to window)
            ax.axhspan(
                max(seq.start_time_year, t_start),
                min(seq.end_time_year, t_end),
                facecolor=color,
                alpha=sequence_alpha,
                zorder=0,
            )

            if label_sequences:
                label_y = (seq.start_time_year + seq.end_time_year) / 2.0
                label_x = x_min_km * 0.97  # near the left edge
                ax.text(
                    label_x,
                    label_y,
                    f"S{seq_id + 1}\nn={len(seq.event_indices)}\n{seq.fault_coverage_fraction:.0%}",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    clip_on=True,
                    zorder=5,
                )

            legend_handles.append(
                mpatches.Patch(
                    facecolor=color,
                    alpha=min(sequence_alpha * 3, 0.8),
                    label=f"Seq {seq_id + 1} ({seq.fault_coverage_fraction:.0%}, {seq.duration_years:.0f} yr)",
                )
            )

        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    # --- Overlay historical record ---
    if hist_cords is not None and hist_mass is not None and best_time is not None:
        hist_min_time = np.min(hist_cords[:, 1])
        # shift: hist_min_time → best_time in simulation years
        time_offset = best_time - hist_min_time

        for idx, (c, m) in enumerate(zip(hist_cords, hist_mass)):
            e_p, t_hist = c
            t_sim = t_hist + time_offset
            if not (t_start <= t_sim <= t_end):
                continue
            extent_km = np.array([-e_p, -e_p - m]) * 1e-3
            ax.plot(
                extent_km,
                np.full_like(extent_km, t_sim),
                color=hist_color,
                alpha=hist_alpha,
                linewidth=hist_linewidth,
                label='Historical' if idx == 0 else '',
                zorder=5,
            )

        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=rupture_color, marker='o', markersize=4,
                           linestyle='none', label='Simulation'),
                plt.Line2D([0], [0], color=hist_color, linewidth=hist_linewidth,
                           label='Historical'),
            ],
            loc='best', fontsize=8, framealpha=0.7,
        )

    if output_path is not None:
        print(f"--> Saving rupture plot to {output_path}")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_ot_rupture_window(
    sim_data: SimulationData,
    shapefile_path: str,
    result: Dict[str, Any],
    hist_cords: np.ndarray,
    hist_mass: np.ndarray,
    ax: Optional[plt.Axes] = None,
    config: Optional[Dict[str, Any]] = None,
    # ── Sim rupture styling ────────────────────────────────────────────────────
    rupture_color: str = 'red',
    rupture_markersize: float = 2.0,
    ploting_mag_threshold: float = 6.5,
    label_mag_threshold: float = 7.0,
    add_rupture_direction: bool = True,
    add_catalog_index: bool = False,
    verbose: bool = False,
    # ── Historical styling ─────────────────────────────────────────────────────
    hist_color: str = 'black',
    hist_linewidth: float = 2.0,
    hist_alpha: float = 0.9,
    # ── OT connection styling ──────────────────────────────────────────────────
    ot_cmap=None,
    ot_threshold_frac: float = 1e-3,
    ot_alpha_min: float = 0.15,
    ot_alpha_max: float = 0.9,
    ot_lw_min: float = 0.5,
    ot_lw_max: float = 2.5,
    norm: Optional[mcolors.Normalize] = None,
    # ── Sequence overlays ──────────────────────────────────────────────────────
    sequences: Optional[List[RuptureSequence]] = None,
    sequence_colors: Optional[List[str]] = None,
    sequence_alpha: float = 0.15,
    label_sequences: bool = True,
    color_sequence_events: bool = True,
    # ── Layout ─────────────────────────────────────────────────────────────────
    time_padding_frac: float = 0.05,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, Optional[mcolors.Normalize]]:
    """
    Combined OT-score + rupture-sequence panel for a single simulation window.

    Overlays three layers on a distance-along-strike × simulation-time axes:

    1. **Simulation ruptures** — spatial dot patterns from :func:`get_rupture_mask`,
       coloured by sequence membership when *sequences* is supplied, with optional
       magnitude labels, catalog indices, hypocenter stars and direction arrows.
    2. **Historical events** — plotted as horizontal bars shifted into simulation
       time so that ``min(hist_cords[:, 1])`` aligns with ``result['best_time']``.
    3. **OT transport connections** — :class:`~matplotlib.collections.LineCollection`
       linking each historical event centre to its matched simulation event centres,
       with line weight and alpha scaled by transported mass.

    The panel title shows the OT score, h*, mass conservation and inversion count
    from *result*.

    Args:
        sim_data:             Loaded simulation data.
        shapefile_path:       Path to the fault-trace shapefile.
        result:               OT result dict with keys ``'sim_idx'``, ``'best_time'``,
                              ``'sim_window'``, ``'sim_mass_window'``, ``'P'``,
                              ``'score'``, ``'mass_cons'``, ``'inversions'``, ``'h*'``.
        hist_cords:           Historical event array shape (N, 2): [east_m, year].
        hist_mass:            Historical rupture lengths in metres, shape (N,).
        ax:                   Existing Axes to draw on; a new figure is created if None.
        config:               Override keys from ``DEFAULT_CONFIG``.
        rupture_color:        Default dot colour for simulation ruptures.
        rupture_markersize:   Marker size for rupture dots.
        ploting_mag_threshold: Minimum Mw to visualise from the catalog.
        label_mag_threshold:  Minimum Mw to add a magnitude label.
        add_rupture_direction: Draw hypocenter stars and direction arrows.
        add_catalog_index:    Annotate each labelled event with its catalog index.
        verbose:              Pass through to :func:`analyze_rupture_direction`.
        hist_color:           Colour for historical event bars.
        hist_linewidth:       Line width for historical bars.
        hist_alpha:           Alpha for historical bars.
        ot_cmap:              Colormap for OT connections (default ``cm.cool``).
        ot_threshold_frac:    Hide connections below this fraction of ``P.max()``.
        ot_alpha_min/max:     Alpha range for OT lines (scaled by weight).
        ot_lw_min/max:        Line-width range for OT lines.
        norm:                 Pre-built :class:`~matplotlib.colors.Normalize` for
                              a shared colorbar across multiple panels.
        sequences:            Rupture sequences to overlay as coloured bands.
        sequence_colors:      Colour cycle for sequence bands.
        sequence_alpha:       Band transparency.
        label_sequences:      Annotate each band with its ID, count and coverage.
        color_sequence_events: Colour sim dots by sequence membership.
        time_padding_frac:    Fractional padding added above/below the time window.
        output_path:          If set, save the figure here at 300 dpi.

    Returns:
        ``(fig, ax, norm)`` — the norm can be passed back in for shared colorbars.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    _ot_cmap = ot_cmap if ot_cmap is not None else cm.cool

    # ── Geometry ───────────────────────────────────────────────────────────────
    mesh_along_strike = project_to_fault_trace(sim_data.coords, shapefile_path)
    x_min_km = -np.max(mesh_along_strike) * 1e-3
    x_max_km = 0.0

    # ── Time window (simulation time) ──────────────────────────────────────────
    sim_times = result['sim_window'][:, 1]
    best_time = result['best_time']
    hist_min_time = np.min(hist_cords[:, 1])
    hist_span = np.max(hist_cords[:, 1]) - hist_min_time
    # shift historical → simulation time
    time_offset = best_time - hist_min_time

    t_start = min(sim_times.min(), hist_min_time + time_offset)
    t_end   = max(sim_times.max(), hist_min_time + time_offset + hist_span)
    padding = (t_end - t_start) * time_padding_frac
    t_start -= padding
    t_end   += padding

    # ── Sequence event colour map ──────────────────────────────────────────────
    _seq_colors = sequence_colors or _SEQUENCE_COLORS
    _event_to_seq: dict = {}
    if sequences:
        for seq_id, seq in enumerate(sequences):
            color = _seq_colors[seq_id % len(_seq_colors)]
            for ev_idx in seq.event_indices:
                _event_to_seq[ev_idx] = (seq_id, color)

    # ── Figure / axes ──────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(cfg['panel_width'], cfg['panel_height'])
        )
    else:
        fig = ax.get_figure()

    ax.set_xlim(x_min_km, x_max_km)
    ax.set_ylim(t_start, t_end)
    ax.set_xlabel("Distance Along Strike [km]")
    ax.set_ylabel("Time [years]")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Title (OT metrics) ─────────────────────────────────────────────────────
    score_str = f"{result['score']:.2f}" if result['score'] < 1e98 else "invalid"
    ax.set_title(
        f"Sim {result.get('sim_idx', '?')} | Score: {score_str} | "
        f"h* = {result['h*']:.2f} km\n"
        f"Mass: {result['mass_cons']:.1f}% | Inv: {result['inversions']:.1f}",
        fontsize=int(cfg['font_size_title']),
        fontweight=cfg['font_weight_title'],
    )

    # ── A. Sequence bands ──────────────────────────────────────────────────────
    if sequences:
        for seq_id, seq in enumerate(sequences):
            if seq.end_time_year < t_start or seq.start_time_year > t_end:
                continue
            color = _seq_colors[seq_id % len(_seq_colors)]
            ax.axhspan(
                max(seq.start_time_year, t_start),
                min(seq.end_time_year, t_end),
                facecolor=color, alpha=sequence_alpha, zorder=0,
            )
            if label_sequences:
                label_y = (seq.start_time_year + seq.end_time_year) / 2.0
                ax.text(
                    x_min_km * 0.97, label_y,
                    f"S{seq_id + 1}\nn={len(seq.event_indices)}\n"
                    f"{seq.fault_coverage_fraction:.0%}",
                    fontsize=7, color=color, fontweight='bold',
                    ha='left', va='center', clip_on=True, zorder=5,
                )

    # ── B. Simulation ruptures from catalog ────────────────────────────────────
    max_time = sim_data.time[-1]
    for idx, event in sim_data.catalog.iterrows():
        if event.Mw < ploting_mag_threshold:
            continue
        if not (t_start <= event['Time_year'] <= t_end):
            continue

        centers, is_rup = get_rupture_mask(
            sim_data, idx, mesh_along_strike,
            num_bins=cfg['rupture_bin_count'],
            slip_threshold=cfg['rupture_threshold'],
        )
        if not np.any(is_rup):
            continue

        rup_locs_km = -centers[is_rup] * 1e-3
        eq_time_yr  = event['Time_year']

        if color_sequence_events and idx in _event_to_seq:
            ev_color = _event_to_seq[idx][1]
        else:
            ev_color = rupture_color

        ax.plot(
            rup_locs_km,
            np.full_like(rup_locs_km, eq_time_yr),
            'o', markersize=rupture_markersize,
            color=ev_color, markeredgecolor=ev_color, markeredgewidth=0.1,
        )

        if event['Mw'] >= label_mag_threshold:
            t_span   = t_end - t_start
            x_span   = x_max_km - x_min_km
            t_offset = t_span * 0.025        # label y-offset, relative to window
            ax.text(
                np.mean(rup_locs_km),
                eq_time_yr + t_offset,
                f"$M_w{event['Mw']:.1f}$",
                fontsize=8, fontweight='bold', ha='center', va='center',
            )
            if add_catalog_index:
                ax.text(
                    np.mean(rup_locs_km) + 0.03 * x_span,
                    eq_time_yr + t_offset * 0.6,
                    f"({idx})", fontsize=7, ha='left', va='center', color='gray',
                )

            if add_rupture_direction:
                metrics = analyze_rupture_direction(
                    sim_data, idx, mesh_along_strike, verbose=verbose
                )
                if metrics and metrics.code != 0:
                    arrow_len  = x_span * 0.06          # ~6 % of x range
                    arrow_props = dict(
                        head_width=t_span * 0.03,       # proportional to time axis
                        head_length=x_span * 0.025,     # proportional to space axis
                        fc='k', ec='k', lw=1,
                        length_includes_head=True,
                    )
                    x_hypo = -metrics.hypocenter_dist * 1e-3
                    y_hypo = metrics.hypocenter_time / (365 * 24 * 60 * 60)
                    if metrics.hypocenter_dist is not None:
                        ax.plot(
                            x_hypo, y_hypo, '*',
                            color='yellow', markersize=10,
                            markeredgecolor='black', zorder=10,
                        )
                    if metrics.code == 1:
                        ax.arrow(x_hypo, y_hypo, -arrow_len, 0, **arrow_props, zorder=10)
                    elif metrics.code == -1:
                        ax.arrow(x_hypo, y_hypo,  arrow_len, 0, **arrow_props, zorder=10)
                    elif metrics.code == 2:
                        ax.arrow(x_hypo, y_hypo, -arrow_len, 0, **arrow_props, zorder=10)
                        ax.arrow(x_hypo, y_hypo,  arrow_len, 0, **arrow_props, zorder=10)

    # ── C. Historical events (shifted into simulation time) ────────────────────
    for h_idx, (c, m) in enumerate(zip(hist_cords, hist_mass)):
        e_p, t_hist = c
        t_plot = t_hist + time_offset
        if not (t_start <= t_plot <= t_end):
            continue
        extent_km = np.array([-e_p, -e_p - m]) * 1e-3
        ax.plot(
            extent_km,
            np.full_like(extent_km, t_plot),
            color=hist_color, alpha=hist_alpha, linewidth=hist_linewidth,
            label='Historical' if h_idx == 0 else '',
            zorder=5,
        )

    # ── D. OT transport connections ────────────────────────────────────────────
    P = result['P']
    threshold = ot_threshold_frac * P.max() if P.max() > 0 else 0.0

    sim_centers  = (-result['sim_window'][:, 0] - result['sim_mass_window'] / 2) * 1e-3
    hist_centers = (-hist_cords[:, 0] - hist_mass / 2) * 1e-3

    segments = []
    weights  = []
    for i in range(len(hist_cords)):
        t_hist_plot = hist_cords[i, 1] + time_offset
        for j in range(len(result['sim_window'])):
            w = P[i, j]
            if w > threshold:
                segments.append([
                    (hist_centers[i], t_hist_plot),
                    (sim_centers[j],  result['sim_window'][j, 1]),
                ])
                weights.append(w)

    used_norm = norm
    if weights:
        weights = np.array(weights)
        if used_norm is None:
            used_norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())

        rgba = _ot_cmap(used_norm(weights))
        w_min, w_max = weights.min(), weights.max()
        norm_w = (weights - w_min) / (w_max - w_min) if w_max > w_min else np.ones_like(weights)
        rgba[:, 3]  = ot_alpha_min + norm_w * (ot_alpha_max - ot_alpha_min)
        linewidths  = ot_lw_min    + norm_w * (ot_lw_max    - ot_lw_min)

        lc = LineCollection(segments, colors=rgba, linewidths=linewidths, zorder=3)
        ax.add_collection(lc)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=rupture_color, marker='o', markersize=4,
                   linestyle='none', label='Simulation'),
        plt.Line2D([0], [0], color=hist_color, linewidth=hist_linewidth,
                   label='Historical'),
    ]
    if sequences:
        for seq_id, seq in enumerate(sequences):
            color = _seq_colors[seq_id % len(_seq_colors)]
            legend_handles.append(mpatches.Patch(
                facecolor=color, alpha=min(sequence_alpha * 3, 0.8),
                label=f"Seq {seq_id + 1} ({seq.fault_coverage_fraction:.0%})",
            ))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.7)

    # ── Save ───────────────────────────────────────────────────────────────────
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax, used_norm
