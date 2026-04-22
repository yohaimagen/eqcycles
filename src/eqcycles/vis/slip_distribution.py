import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union

from eqcycles.core.data import SimulationData
from eqcycles.analysis.geometry import compute_along_strike_profile
from eqcycles.analysis.sequences import RuptureSequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_indices(item) -> List[int]:
    """Return a flat list of catalog integer indices from an int, list of ints, or RuptureSequence."""
    if isinstance(item, RuptureSequence):
        return list(item.event_indices)
    elif isinstance(item, int):
        return [item]
    else:
        out = []
        for sub in item:
            out.extend(_resolve_indices(sub))
        return sorted(set(out))


# ---------------------------------------------------------------------------
# Step 2 — Public plotting primitive
# ---------------------------------------------------------------------------

def plot_along_strike_profile(
    ax: plt.Axes,
    bin_centers_km: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    color: str = "#2166ac",
    fill_alpha: float = 0.25,
    linewidth: float = 1.5,
    label: Optional[str] = None,
    alpha: float = 1.0,
) -> None:
    """
    Draws a single along-strike profile (line + shaded fill) on *ax*.

    This is the low-level building block used by all higher-level slip
    distribution plots.  It is public so that users can call
    ``compute_along_strike_profile`` + ``plot_along_strike_profile`` directly
    to assemble fully custom panels without reimplementing the binning logic.

    Args:
        ax: Matplotlib axes to draw on.
        bin_centers_km: Along-strike bin centres (km), e.g. from
            ``compute_along_strike_profile``.
        values: Per-bin scalar values to plot.
        valid: Boolean mask; only ``True`` bins are drawn.
        color: Line and fill colour.
        fill_alpha: Transparency of the shaded area beneath the profile.
        linewidth: Width of the profile line.
        label: Legend label (passed to ``ax.plot``).
        alpha: Overall line opacity.
    """
    x = bin_centers_km[valid]
    y = values[valid]
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)
    ax.fill_between(x, y, alpha=fill_alpha, color=color)


# ---------------------------------------------------------------------------
# Step 3 — Refactored public functions
# ---------------------------------------------------------------------------

def plot_slip_distribution(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    event_indices: Union[int, List[int], RuptureSequence],
    ax: plt.Axes = None,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    color: str = "#2166ac",
    fill_alpha: float = 0.25,
    linewidth: float = 1.5,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plots the cumulative slip distribution along the fault for one or more events.

    Nodes deeper than ``max_depth_km`` are excluded before averaging.  For each
    along-strike bin the slip values of the remaining shallow nodes are averaged
    across depth, then summed across all requested events.  The result is plotted
    as cumulative slip (y-axis, metres) vs. distance along strike (x-axis, km).

    Args:
        sim_data: The standardized simulation data object.  ``sim_data.eq_slip``
            must be present (shape: n_nodes × n_events).
        mesh_along_strike: Along-strike distance (metres) for each mesh node,
            as returned by ``project_to_fault_trace``.
        event_indices: One of:
            - a single catalog integer index,
            - a list of catalog integer indices,
            - a :class:`~eqcycles.analysis.sequences.RuptureSequence` object
              (its ``event_indices`` list is used automatically).
        ax: Existing Matplotlib axes to draw on.  If None a new figure is created.
        num_bins: Number of along-strike bins used to aggregate node slip values.
        max_depth_km: Only nodes shallower than this depth (km) are included in
            the along-strike average.
        color: Line and fill colour.
        fill_alpha: Transparency of the area fill beneath the slip profile.
        linewidth: Width of the slip profile line.
        title: Axes title.  Auto-generated if None.
        output_path: If given, the figure is saved to this path at 300 dpi.

    Returns:
        ``(fig, ax)`` — the Matplotlib figure and axes objects.
    """
    if sim_data.eq_slip is None:
        raise ValueError("sim_data.eq_slip is None — cannot compute slip distribution.")

    if isinstance(event_indices, RuptureSequence):
        auto_title = (
            f"Cumulative slip — sequence of {len(event_indices.event_indices)} event(s), "
            f"{event_indices.fault_coverage_fraction:.0%} fault coverage"
        )
    elif isinstance(event_indices, int):
        row = sim_data.catalog.iloc[event_indices]
        auto_title = f"Slip distribution — event {event_indices} (Mw {row.Mw:.1f})"
    else:
        auto_title = "Cumulative slip — combined events"

    indices = _resolve_indices(event_indices)
    node_slip = sim_data.eq_slip[:, indices].sum(axis=1)

    bin_centers_km, profile, valid = compute_along_strike_profile(
        mesh_along_strike, node_slip, sim_data.coords, num_bins, max_depth_km
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    plot_along_strike_profile(ax, bin_centers_km, profile, valid, color=color,
                              fill_alpha=fill_alpha, linewidth=linewidth)

    ax.set_xlabel("Distance Along Strike [km]")
    ax.set_ylabel("Cumulative Slip [m]")
    ax.set_title(title or auto_title, fontsize=11, fontweight="bold")
    ax.set_xlim(bin_centers_km[valid].min(), bin_centers_km[valid].max())
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.99, 0.97, f"d ≤ {max_depth_km:.0f} km",
            transform=ax.transAxes, fontsize=7, ha="right", va="top", color="gray")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_slip_distributions_overlay(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    sequences: List[RuptureSequence],
    ax: plt.Axes = None,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    colors: Optional[List[str]] = None,
    fill_alpha: float = 0.15,
    line_alpha: float = 0.8,
    linewidth: float = 1.5,
    title: Optional[str] = None,
    show_legend: bool = True,
    show_sum: bool = True,
    sum_color: str = "black",
    sum_alpha: float = 0.12,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plots the slip distribution of multiple sequences on the same axes.

    Each sequence is drawn with a distinct colour.  The fill transparency is
    kept low so overlapping profiles remain readable.

    Args:
        sim_data: The standardized simulation data object.
        mesh_along_strike: Along-strike distances (metres) for each mesh node.
        sequences: List of :class:`~eqcycles.analysis.sequences.RuptureSequence`
            objects to overlay.
        ax: Existing Matplotlib axes.  If None a new figure is created.
        num_bins: Number of along-strike bins per profile.
        max_depth_km: Depth cut-off (km) for the node average.
        colors: List of colours to cycle through.  Defaults to the current
            Matplotlib colour cycle.
        fill_alpha: Transparency of the filled area under each profile.
        line_alpha: Transparency of the profile line itself.
        linewidth: Line width for each profile.
        title: Axes title.  Defaults to "Cumulative slip — N sequences".
        show_legend: If True, add a legend with per-sequence labels.
        show_sum: If True (default), add a right y-axis showing the bin-wise
            sum across all plotted sequences as a black filled profile.
        sum_color: Colour for the summed profile.
        sum_alpha: Fill transparency for the summed profile.
        output_path: If given, the figure is saved to this path at 300 dpi.

    Returns:
        ``(fig, ax)`` — the Matplotlib figure and axes objects.
    """
    if not sequences:
        raise ValueError("No sequences provided.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    _colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Bin geometry is shared across sequences — compute once from first profile
    # to obtain bin_centers_km and valid; reuse for accumulator.
    first_indices = _resolve_indices(sequences[0])
    first_slip = sim_data.eq_slip[:, first_indices].sum(axis=1)
    bin_centers_km, _, valid = compute_along_strike_profile(
        mesh_along_strike, first_slip, sim_data.coords, num_bins, max_depth_km
    )
    sum_slip = np.zeros(len(bin_centers_km))

    for i, item in enumerate(sequences):
        color = _colors[i % len(_colors)]
        indices = _resolve_indices(item)
        node_slip = sim_data.eq_slip[:, indices].sum(axis=1)

        _, profile, valid_i = compute_along_strike_profile(
            mesh_along_strike, node_slip, sim_data.coords, num_bins, max_depth_km
        )
        sum_slip += profile

        if isinstance(item, RuptureSequence):
            label = (
                f"Seq {i + 1}  n={len(indices)}"
                f"  {item.fault_coverage_fraction:.0%}"
                f"  {item.duration_years:.0f} yr"
            )
        else:
            label = f"Group {i + 1}  n={len(indices)}"

        plot_along_strike_profile(ax, bin_centers_km, profile, valid_i,
                                  color=color, fill_alpha=fill_alpha,
                                  linewidth=linewidth, label=label, alpha=line_alpha)

    ax.set_xlabel("Distance Along Strike [km]")
    ax.set_ylabel("Cumulative Slip per Sequence [m]")
    ax.set_title(title or f"Cumulative slip — {len(sequences)} sequences",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.text(0.99, 0.97, f"d ≤ {max_depth_km:.0f} km",
            transform=ax.transAxes, fontsize=7, ha="right", va="top", color="gray")

    if show_sum:
        ax2 = ax.twinx()
        plot_along_strike_profile(ax2, bin_centers_km, sum_slip, valid,
                                  color=sum_color, fill_alpha=sum_alpha,
                                  linewidth=linewidth + 0.5, alpha=0.9,
                                  label="Sum (all sequences)")
        ax2.set_ylabel("Total Cumulative Slip [m]", color=sum_color)
        ax2.tick_params(axis="y", labelcolor=sum_color)
        ax2.set_ylim(bottom=0)
        ax2.spines["top"].set_visible(False)

    if show_legend:
        ax.legend(fontsize=7, framealpha=0.7, loc="upper left")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_slip_distribution_time_window(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    t_start: float,
    t_end: float,
    ax: plt.Axes = None,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    min_mw: float = 0.0,
    slip_color: str = "#2166ac",
    fill_alpha: float = 0.2,
    linewidth: float = 1.5,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plots the cumulative slip and seismic slip rate for all events within a
    time window, with two y-axes sharing the along-strike x-axis.

    Left y-axis:  cumulative co-seismic slip (m) summed over all events whose
                  ``Time_year`` falls within [t_start, t_end].
    Right y-axis: seismic slip rate (mm/yr) = cumulative slip / (t_end - t_start).

    Nodes deeper than ``max_depth_km`` are excluded; the remaining shallow nodes
    are averaged across depth within each along-strike bin.

    Args:
        sim_data: The standardized simulation data object.
        mesh_along_strike: Along-strike distances (metres) for each mesh node.
        t_start: Start of the time window (years).
        t_end: End of the time window (years).
        ax: Existing Matplotlib axes for the left y-axis.  If None a new figure
            is created.
        num_bins: Number of along-strike bins.
        max_depth_km: Depth cut-off (km) for the shallow-node average.
        min_mw: Minimum magnitude; events below this are excluded.
        slip_color: Colour for both profiles (left and right axes).
        fill_alpha: Fill transparency for both profiles.
        linewidth: Line width for both profiles.
        title: Axes title.  Auto-generated if None.
        output_path: If given, the figure is saved to this path at 300 dpi.

    Returns:
        ``(fig, ax_slip, ax_rate)`` — figure, left axes, right axes.
    """
    time_span = t_end - t_start
    if time_span <= 0:
        raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start}).")

    if sim_data.eq_slip is None:
        raise ValueError("sim_data.eq_slip is None — cannot compute slip distribution.")

    cat = sim_data.catalog
    mask_time = (cat["Time_year"] >= t_start) & (cat["Time_year"] <= t_end)
    if min_mw > 0:
        mask_time &= cat["Mw"] >= min_mw
    indices = cat.index[mask_time].tolist()

    if not indices:
        raise ValueError(
            f"No events found in [{t_start}, {t_end}] yr"
            + (f" with Mw ≥ {min_mw}" if min_mw > 0 else "") + "."
        )

    node_slip = sim_data.eq_slip[:, indices].sum(axis=1)
    bin_centers_km, slip_profile, valid = compute_along_strike_profile(
        mesh_along_strike, node_slip, sim_data.coords, num_bins, max_depth_km
    )
    rate_profile = slip_profile * 1000.0 / time_span  # mm/yr

    if ax is None:
        fig, ax_slip = plt.subplots(figsize=(10, 4))
    else:
        ax_slip = ax
        fig = ax_slip.get_figure()

    ax_rate = ax_slip.twinx()

    plot_along_strike_profile(ax_slip, bin_centers_km, slip_profile, valid,
                              color=slip_color, fill_alpha=fill_alpha, linewidth=linewidth)
    plot_along_strike_profile(ax_rate, bin_centers_km, rate_profile, valid,
                              color=slip_color, fill_alpha=fill_alpha, linewidth=linewidth)

    ax_slip.set_xlabel("Distance Along Strike [km]")
    ax_slip.set_ylabel("Cumulative Slip [m]", color=slip_color)
    ax_slip.tick_params(axis="y", labelcolor=slip_color)
    ax_slip.set_ylim(bottom=0)
    ax_slip.spines["top"].set_visible(False)

    ax_rate.set_ylabel("Seismic Slip Rate [mm/yr]", color=slip_color)
    ax_rate.tick_params(axis="y", labelcolor=slip_color)
    ax_rate.set_ylim(bottom=0)
    ax_rate.spines["top"].set_visible(False)

    auto_title = (
        f"Slip distribution  |  {t_start:.0f} – {t_end:.0f} yr"
        f"  |  n={len(indices)} events"
        + (f"  |  Mw ≥ {min_mw}" if min_mw > 0 else "")
    )
    ax_slip.set_title(title or auto_title, fontsize=11, fontweight="bold")
    ax_slip.text(0.99, 0.97, f"d ≤ {max_depth_km:.0f} km",
                 transform=ax_slip.transAxes, fontsize=7, ha="right", va="top", color="gray")

    handles = [
        plt.Line2D([0], [0], color=slip_color, linewidth=linewidth,
                   label="Cumulative slip [m]"),
        plt.Line2D([0], [0], color=slip_color, linewidth=linewidth, linestyle="--",
                   label="Slip rate [mm/yr]"),
    ]
    ax_slip.legend(handles=handles, fontsize=7, framealpha=0.7, loc="upper left")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax_slip, ax_rate


def plot_slip_distributions_comparison(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    sequences: List[RuptureSequence],
    ncols: int = 2,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plots one slip distribution panel per sequence in a grid layout.

    Args:
        sim_data: The standardized simulation data object.
        mesh_along_strike: Along-strike distances (metres) for each mesh node.
        sequences: List of :class:`~eqcycles.analysis.sequences.RuptureSequence`
            objects, e.g. from ``find_rupture_sequences``.
        ncols: Number of columns in the subplot grid.
        num_bins: Number of along-strike bins per panel.
        max_depth_km: Depth cut-off (km) passed to each panel.
        output_path: If given, the figure is saved to this path at 300 dpi.

    Returns:
        ``(fig, axs)`` — the figure and flattened axes array.
    """
    n = len(sequences)
    if n == 0:
        raise ValueError("No sequences provided.")

    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), sharey=False)
    axs_flat = np.array(axs).flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, seq in enumerate(sequences):
        color = colors[i % len(colors)]
        plot_slip_distribution(
            sim_data,
            mesh_along_strike,
            seq,
            ax=axs_flat[i],
            num_bins=num_bins,
            max_depth_km=max_depth_km,
            color=color,
            title=(
                f"Seq {i + 1}  |  n={len(seq.event_indices)}  |  "
                f"{seq.fault_coverage_fraction:.0%}  |  {seq.duration_years:.0f} yr"
            ),
        )

    for j in range(n, len(axs_flat)):
        axs_flat[j].set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axs_flat


# ---------------------------------------------------------------------------
# Step 4 — Generic input/output comparison
# ---------------------------------------------------------------------------

def plot_along_strike_comparison(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    left_field: np.ndarray,
    right_field: np.ndarray,
    left_label: str = "Left field",
    right_label: str = "Right field",
    left_ylabel: Optional[str] = None,
    right_ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    left_color: str = "#2166ac",
    right_color: str = "#d6604d",
    fill_alpha: float = 0.2,
    linewidth: float = 1.5,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plots two along-strike profiles on the same axes with twin y-axes.

    The left and right fields can represent any two node-level quantities
    derived from ``sim_data`` — e.g. co-seismic slip vs. total accumulated slip,
    or seismic slip vs. time-mean slip rate.  The caller is responsible for
    extracting and reducing the fields before calling this function:

    Examples::

        # Co-seismic slip (output) vs. total slip over same window (input)
        seismic = sim_data.eq_slip[:, indices].sum(axis=1)
        total   = sim_data.slip[:, t_end_idx] - sim_data.slip[:, t_start_idx]
        plot_along_strike_comparison(sim_data, mesh, seismic, total, ...)

        # Seismic slip vs. time-mean log10-velocity (slip rate proxy)
        seismic   = sim_data.eq_slip[:, indices].sum(axis=1)
        mean_logv = sim_data.slip_rate.mean(axis=1)
        plot_along_strike_comparison(sim_data, mesh, seismic, mean_logv, ...)

    Args:
        sim_data: The standardized simulation data object (used only for
            ``coords`` in the depth filter).
        mesh_along_strike: Along-strike distance (metres) for each mesh node.
        left_field: Per-node values ``(n_nodes,)`` plotted on the left y-axis.
        right_field: Per-node values ``(n_nodes,)`` plotted on the right y-axis.
        left_label: Legend label for the left profile.
        right_label: Legend label for the right profile.
        left_ylabel: Left y-axis label.  Defaults to ``left_label``.
        right_ylabel: Right y-axis label.  Defaults to ``right_label``.
        ax: Existing Matplotlib axes for the left y-axis.  If None a new figure
            is created.
        num_bins: Number of along-strike bins.
        max_depth_km: Depth cut-off (km) for the shallow-node average.
        left_color: Colour for the left profile.
        right_color: Colour for the right profile.
        fill_alpha: Fill transparency for both profiles.
        linewidth: Line width for both profiles.
        title: Axes title.
        output_path: If given, the figure is saved to this path at 300 dpi.

    Returns:
        ``(fig, ax_left, ax_right)``
    """
    bin_centers_km, left_profile, left_valid = compute_along_strike_profile(
        mesh_along_strike, left_field, sim_data.coords, num_bins, max_depth_km
    )
    _, right_profile, right_valid = compute_along_strike_profile(
        mesh_along_strike, right_field, sim_data.coords, num_bins, max_depth_km
    )

    if ax is None:
        fig, ax_left = plt.subplots(figsize=(10, 4))
    else:
        ax_left = ax
        fig = ax_left.get_figure()

    ax_right = ax_left.twinx()

    plot_along_strike_profile(ax_left, bin_centers_km, left_profile, left_valid,
                              color=left_color, fill_alpha=fill_alpha,
                              linewidth=linewidth, label=left_label)
    plot_along_strike_profile(ax_right, bin_centers_km, right_profile, right_valid,
                              color=right_color, fill_alpha=fill_alpha,
                              linewidth=linewidth, label=right_label)

    ax_left.set_xlabel("Distance Along Strike [km]")
    ax_left.set_ylabel(left_ylabel or left_label, color=left_color)
    ax_left.tick_params(axis="y", labelcolor=left_color)
    ax_left.set_ylim(bottom=0)
    ax_left.spines["top"].set_visible(False)

    ax_right.set_ylabel(right_ylabel or right_label, color=right_color)
    ax_right.tick_params(axis="y", labelcolor=right_color)
    ax_right.set_ylim(bottom=0)
    ax_right.spines["top"].set_visible(False)

    ax_left.text(0.99, 0.97, f"d ≤ {max_depth_km:.0f} km",
                 transform=ax_left.transAxes, fontsize=7, ha="right", va="top", color="gray")

    if title:
        ax_left.set_title(title, fontsize=11, fontweight="bold")

    # Combined legend
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right,
                   fontsize=7, framealpha=0.7, loc="upper left")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax_left, ax_right


# ---------------------------------------------------------------------------
# Step 5 — Grid comparison: one panel per sequence
# ---------------------------------------------------------------------------

def plot_input_output_comparison(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    sequences: List[RuptureSequence],
    reference_field: np.ndarray,
    reference_label: str = "Reference",
    reference_ylabel: Optional[str] = None,
    ncols: int = 2,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    seismic_color: str = "#2166ac",
    reference_color: str = "#d6604d",
    output_path: Optional[str] = None,
) -> tuple:
    """
    Grid of comparison panels — one per sequence — each showing co-seismic slip
    (left y-axis) against a shared reference field (right y-axis).

    Typical use: pass the time-mean slip rate or the total aseismic slip as
    ``reference_field`` to compare it against the seismic contribution from each
    rupture sequence.

    Args:
        sim_data: The standardized simulation data object.
        mesh_along_strike: Along-strike distances (metres) for each mesh node.
        sequences: List of :class:`~eqcycles.analysis.sequences.RuptureSequence`
            objects, e.g. from ``find_rupture_sequences``.
        reference_field: Per-node array ``(n_nodes,)`` used as the right-axis
            profile in every panel.  Computed once and reused across all panels.
        reference_label: Legend label for the reference profile.
        reference_ylabel: Right y-axis label.  Defaults to ``reference_label``.
        ncols: Number of columns in the subplot grid.
        num_bins: Number of along-strike bins per panel.
        max_depth_km: Depth cut-off (km) for the shallow-node average.
        seismic_color: Colour for the co-seismic slip (left) profile.
        reference_color: Colour for the reference (right) profile.
        output_path: If given, the figure is saved to this path at 300 dpi.

    Returns:
        ``(fig, axs_flat)`` — the figure and flattened left-axes array.
    """
    n = len(sequences)
    if n == 0:
        raise ValueError("No sequences provided.")

    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), sharey=False)
    axs_flat = np.array(axs).flatten()

    for i, seq in enumerate(sequences):
        seismic_field = sim_data.eq_slip[:, seq.event_indices].sum(axis=1)
        plot_along_strike_comparison(
            sim_data,
            mesh_along_strike,
            left_field=seismic_field,
            right_field=reference_field,
            left_label="Co-seismic slip",
            right_label=reference_label,
            left_ylabel="Cumulative Slip [m]",
            right_ylabel=reference_ylabel or reference_label,
            ax=axs_flat[i],
            num_bins=num_bins,
            max_depth_km=max_depth_km,
            left_color=seismic_color,
            right_color=reference_color,
            title=(
                f"Seq {i + 1}  |  n={len(seq.event_indices)}  |  "
                f"{seq.fault_coverage_fraction:.0%}  |  {seq.duration_years:.0f} yr"
            ),
        )

    for j in range(n, len(axs_flat)):
        axs_flat[j].set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axs_flat
