"""
Visualization of Optimal Transport (OT) mappings between historical and simulated
earthquake catalogs.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from typing import Dict, Any, Optional


def _collect_ot_weights(
    result: Dict[str, Any],
    ot_threshold_frac: float = 1e-3,
) -> np.ndarray:
    """Return the above-threshold OT weights from a single result dict."""
    P = result['P']
    if P.max() == 0:
        return np.array([])
    threshold = ot_threshold_frac * P.max()
    w = P[P > threshold]
    return w


def plot_ot_window(
    ax: plt.Axes,
    result: Dict[str, Any],
    hist_cords: np.ndarray,
    hist_mass: np.ndarray,
    is_first_column: bool = False,
    sim_color: str = 'gray',
    hist_color: str = 'black',
    ot_cmap=cm.cool,
    sim_linewidth: float = 4.0,
    hist_linewidth: float = 2.0,
    sim_alpha: float = 0.6,
    hist_alpha: float = 0.8,
    ot_alpha_min: float = 0.15,
    ot_alpha_max: float = 1.0,
    ot_lw_min: float = 0.5,
    ot_lw_max: float = 2.5,
    ot_threshold_frac: float = 1e-3,
    norm: Optional[mcolors.Normalize] = None,
) -> Optional[mcolors.Normalize]:
    """
    Plot a single OT mapping panel.

    Time axis is in **historical record years**. Simulation events are shifted
    so that ``best_time`` aligns with the start of the historical catalog.

    Parameters
    ----------
    ax : plt.Axes
    result : dict
        Keys: 'sim_idx', 'best_time', 'sim_window', 'sim_mass_window',
        'P', 'score', 'mass_cons', 'inversions'.
    hist_cords : np.ndarray, shape (N, 2)
        [east_position_m, historical_year].
    hist_mass : np.ndarray, shape (N,)
        Rupture lengths in metres.
    is_first_column : bool
        Add y-axis label when True.
    sim_color, hist_color : str
        Line colours for simulation / historical events.
    ot_cmap : colormap
    sim_linewidth, hist_linewidth : float
    sim_alpha, hist_alpha : float
    ot_alpha_min, ot_alpha_max : float
        Dynamic alpha range for OT connection lines.
    ot_lw_min, ot_lw_max : float
        Dynamic linewidth range for OT connection lines.
    ot_threshold_frac : float
        Connections below ``ot_threshold_frac * P.max()`` are hidden.
    norm : mcolors.Normalize, optional
        Pre-built colour norm (shared across panels). If None, a local norm
        is derived from the weights in this panel only.

    Returns
    -------
    norm : mcolors.Normalize or None
        The norm used (local or passed-in), so the caller can attach a
        shared colorbar.
    """
    best_time = result['best_time']
    sim_window = result['sim_window']        # (M, 2): [east_m, sim_time_yr]
    sim_mass_window = result['sim_mass_window']
    P = result['P']

    hist_min_time = np.min(hist_cords[:, 1])

    # Event centres along strike [km], positive = east convention flipped
    sim_centers = (-sim_window[:, 0] - sim_mass_window / 2) * 1e-3
    hist_centers = (-hist_cords[:, 0] - hist_mass / 2) * 1e-3

    # Time offset to convert simulation time → historical years
    # sim_time=best_time corresponds to hist_min_time
    sim_to_hist = hist_min_time - best_time

    # --- A. Simulation events (gray), shifted to historical time ---
    for idx, (c, m) in enumerate(zip(sim_window, sim_mass_window)):
        e_p, t_sim = c
        extent = np.array([-e_p, -e_p - m]) * 1e-3
        ax.plot(
            extent,
            np.full_like(extent, t_sim + sim_to_hist),
            color=sim_color,
            alpha=sim_alpha,
            linewidth=sim_linewidth,
            label='Simulation' if idx == 0 else '',
        )

    # --- B. Historical events (black) at their actual years ---
    for idx, (c, m) in enumerate(zip(hist_cords, hist_mass)):
        e_p, t_hist = c
        extent = np.array([-e_p, -e_p - m]) * 1e-3
        ax.plot(
            extent,
            np.full_like(extent, t_hist),
            color=hist_color,
            alpha=hist_alpha,
            linewidth=hist_linewidth,
            label='Historical' if idx == 0 else '',
        )

    # --- C. OT connections with dynamic alpha & linewidth ---
    threshold = ot_threshold_frac * P.max() if P.max() > 0 else 0.0
    segments = []
    weights = []

    for i in range(len(hist_cords)):
        t_hist = hist_cords[i, 1]
        for j in range(len(sim_window)):
            weight = P[i, j]
            if weight > threshold:
                segments.append([
                    (hist_centers[i], t_hist),
                    (sim_centers[j], sim_window[j, 1] + sim_to_hist),
                ])
                weights.append(weight)

    used_norm = norm
    if weights:
        weights = np.array(weights)

        if used_norm is None:
            used_norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())

        rgba_colors = ot_cmap(used_norm(weights))

        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            norm_w = (weights - w_min) / (w_max - w_min)
        else:
            norm_w = np.ones_like(weights)

        rgba_colors[:, 3] = ot_alpha_min + norm_w * (ot_alpha_max - ot_alpha_min)
        linewidths = ot_lw_min + norm_w * (ot_lw_max - ot_lw_min)

        lc = LineCollection(segments, colors=rgba_colors, linewidths=linewidths, zorder=10)
        ax.add_collection(lc)

    # --- Formatting ---
    score_str = f"{result['score']:.2f}" if result['score'] < 1e98 else "1e99 (Invalid)"
    ax.set_title(
        f"Sim {result['sim_idx']} | Score: {score_str} | h* = {result['h*']:.2f} km\n"
        f"Mass: {result['mass_cons']:.1f}% | Inversions: {result['inversions']:.1f}",
        fontsize=10,
        fontweight='bold',
    )
    ax.set_xlabel("Dist. Along Strike [km]", fontsize=8)
    if is_first_column:
        ax.set_ylabel("Time [years]")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.autoscale_view()

    return used_norm


def plot_ranked_ot_windows(
    results_list: list,
    hist_cords: np.ndarray,
    hist_mass: np.ndarray,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    panel_width: float = 4.0,
    panel_height: float = 6.0,
    ot_cmap=cm.cool,
    ot_threshold_frac: float = 1e-3,
    **plot_kwargs,
) -> plt.Figure:
    """
    Plot a row of OT mapping panels with a **single shared colorbar** on the right.

    A global colour norm is computed from all OT weights in the batch so every
    panel uses the same scale.  Time is expressed in historical record years;
    simulation events are shifted to align with the historical catalog.

    Parameters
    ----------
    results_list : list
        Result dicts sorted best-first.
    hist_cords : np.ndarray
        Historical event coordinates [east_m, year].
    hist_mass : np.ndarray
        Historical rupture lengths [m].
    start_idx : int
        First index in results_list to plot.
    end_idx : int, optional
        One-past-last index (default: start_idx + 5).
    panel_width, panel_height : float
        Size of each panel in inches.
    ot_cmap : colormap
        Shared colormap for OT connections.
    ot_threshold_frac : float
        Threshold fraction forwarded to each panel.
    **plot_kwargs
        Extra keyword arguments forwarded to :func:`plot_ot_window`
        (except ``norm`` and ``ot_cmap``, which are managed here).

    Returns
    -------
    fig : plt.Figure
    """
    if end_idx is None:
        end_idx = min(start_idx + 5, len(results_list))

    batch = results_list[start_idx:end_idx]
    num = len(batch)

    # --- Pass 1: collect all weights to build a global norm ---
    all_weights = np.concatenate([
        _collect_ot_weights(r, ot_threshold_frac) for r in batch
    ])
    if len(all_weights) > 0:
        global_norm = mcolors.Normalize(vmin=all_weights.min(), vmax=all_weights.max())
    else:
        global_norm = mcolors.Normalize(vmin=0, vmax=1)

    # --- Build figure with extra room on the right for the colorbar ---
    fig, axes = plt.subplots(1, num, figsize=(panel_width * num, panel_height), sharey=False)
    if num == 1:
        axes = [axes]

    # --- Pass 2: draw each panel ---
    for plot_i, result in enumerate(batch):
        plot_ot_window(
            axes[plot_i],
            result,
            hist_cords,
            hist_mass,
            is_first_column=(plot_i == 0),
            ot_cmap=ot_cmap,
            ot_threshold_frac=ot_threshold_frac,
            norm=global_norm,
            **plot_kwargs,
        )

    # --- Legend on the first axes only ---
    sim_color = plot_kwargs.get('sim_color', 'gray')
    hist_color = plot_kwargs.get('hist_color', 'black')
    legend_handles = [
        plt.Line2D([0], [0], color=hist_color, linewidth=2, label='Historical'),
        plt.Line2D([0], [0], color=sim_color, linewidth=2, label='Simulation'),
    ]
    axes[0].legend(handles=legend_handles, loc='best', fontsize=8, framealpha=0.7)

    fig.tight_layout()

    # --- Single shared colorbar, attached only to the rightmost axes ---
    sm = plt.cm.ScalarMappable(cmap=ot_cmap, norm=global_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Mass Transported', fontsize=10)
    return fig
