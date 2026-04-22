from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from eqcycles.core.data import SimulationData
from eqcycles.analysis.rupture import get_rupture_mask


@dataclass
class RuptureSequence:
    """
    Stores the events and metrics for a single detected rupture sequence —
    a group of earthquakes that together cover the full (or near-full) fault length.
    """
    event_indices: List[int]        # Catalog indices of all member events
    start_time_year: float          # Time of the earliest event in the sequence
    end_time_year: float            # Time of the last event that completed coverage
    duration_years: float           # end_time_year - start_time_year
    fault_coverage_fraction: float  # Fraction of fault bins covered (0.0–1.0)
    covered_distance_m: float       # Total fault length covered (meters)
    total_fault_length_m: float     # Total fault length used as reference (meters)
    segment_coverage: np.ndarray = field(repr=False)  # Boolean (num_segments,) mask


def compute_event_coverage(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    slip_threshold: float = 0.05,
    num_segments: int = 100,
    min_mw: float = 6.5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Pre-computes, for each qualifying event, which fault segments it covers.

    Args:
        sim_data: The standardized simulation data object.
        mesh_along_strike: Along-strike distance (meters) for each mesh node,
            as returned by ``project_to_fault_trace``.
        slip_threshold: Slip (meters) above which a bin is considered ruptured.
        num_segments: Number of along-strike bins to divide the fault into.
        min_mw: Minimum magnitude; events below this threshold are skipped.

    Returns:
        A tuple (coverage_df, segment_masks):
        - coverage_df: DataFrame with columns ``event_idx``, ``time_year``, ``Mw``,
          ``min_dist_m``, ``max_dist_m``, ``rupture_len_m``.  One row per
          qualifying event that has non-zero rupture coverage.
        - segment_masks: Boolean array of shape (n_qualifying_events, num_segments).
          ``segment_masks[i, k]`` is True when event i ruptured bin k.
    """
    rows = []
    masks = []

    for idx, event in sim_data.catalog.iterrows():
        if event.Mw < min_mw:
            continue

        bin_centers, is_ruptured = get_rupture_mask(
            sim_data, idx, mesh_along_strike, num_segments, slip_threshold
        )

        if not np.any(is_ruptured):
            continue

        ruptured_centers = bin_centers[is_ruptured]
        min_dist = float(ruptured_centers.min())
        max_dist = float(ruptured_centers.max())

        rows.append({
            "event_idx": int(idx),
            "time_year": float(event.Time_year),
            "Mw": float(event.Mw),
            "min_dist_m": min_dist,
            "max_dist_m": max_dist,
            "rupture_len_m": max_dist - min_dist,
        })
        masks.append(is_ruptured)

    if not rows:
        coverage_df = pd.DataFrame(
            columns=["event_idx", "time_year", "Mw", "min_dist_m", "max_dist_m", "rupture_len_m"]
        )
        return coverage_df, np.empty((0, num_segments), dtype=bool)

    coverage_df = pd.DataFrame(rows).reset_index(drop=True)
    segment_masks = np.array(masks, dtype=bool)  # shape (n_events, num_segments)

    return coverage_df, segment_masks


def find_rupture_sequences(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    slip_threshold: float = 0.05,
    num_segments: int = 100,
    coverage_threshold: float = 0.9,
    min_mw: float = 6.5,
) -> List[RuptureSequence]:
    """
    Finds all rupture sequences in the simulation catalog using a greedy
    single-pass accumulation.

    Starting from a blank coverage state, events are consumed chronologically.
    Each event's ruptured bins are OR-ed into the running coverage.  The moment
    coverage reaches ``coverage_threshold`` a sequence is recorded and the
    coverage slate is wiped clean — the next event starts a fresh accumulation.
    There is no time-window limit: a sequence simply ends when the fault is
    covered, however long that takes.

    Args:
        sim_data: The standardized simulation data object.
        mesh_along_strike: Along-strike distances (meters) for each mesh node.
        slip_threshold: Slip (m) above which a bin is considered ruptured.
        num_segments: Number of along-strike bins to divide the fault into.
        coverage_threshold: Fraction of bins that must be covered to close a
            sequence (0–1).  E.g. 0.9 = 90 % of the fault length.
        min_mw: Minimum magnitude for events considered in the sweep.

    Returns:
        A list of :class:`RuptureSequence` objects in chronological order.
    """
    coverage_df, segment_masks = compute_event_coverage(
        sim_data, mesh_along_strike, slip_threshold, num_segments, min_mw
    )

    if coverage_df.empty:
        return []

    # Ensure chronological order
    sort_order = coverage_df["time_year"].argsort().values
    coverage_df = coverage_df.iloc[sort_order].reset_index(drop=True)
    segment_masks = segment_masks[sort_order]

    times = coverage_df["time_year"].values
    n_events = len(times)

    # Fault length geometry (consistent with get_rupture_mask bin edges)
    max_dist = float(mesh_along_strike.max())
    bin_edges = np.linspace(0, max_dist, num_segments + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_fault_length_m = float(bin_centers[-1] - bin_centers[0])
    bin_width_m = total_fault_length_m / (num_segments - 1) if num_segments > 1 else total_fault_length_m

    required_bins = int(np.ceil(coverage_threshold * num_segments))

    sequences: List[RuptureSequence] = []

    # Greedy single-pass accumulation
    accumulated = np.zeros(num_segments, dtype=bool)
    seq_local_indices: List[int] = []

    for j in range(n_events):
        accumulated |= segment_masks[j]
        seq_local_indices.append(j)

        if accumulated.sum() >= required_bins:
            covered_bins = int(accumulated.sum())
            catalog_event_indices = coverage_df.iloc[seq_local_indices]["event_idx"].tolist()

            sequences.append(RuptureSequence(
                event_indices=catalog_event_indices,
                start_time_year=float(times[seq_local_indices[0]]),
                end_time_year=float(times[j]),
                duration_years=float(times[j] - times[seq_local_indices[0]]),
                fault_coverage_fraction=float(covered_bins / num_segments),
                covered_distance_m=float(covered_bins * bin_width_m),
                total_fault_length_m=float(total_fault_length_m),
                segment_coverage=accumulated.copy(),
            ))

            # Reset — next event starts a blank slate
            accumulated = np.zeros(num_segments, dtype=bool)
            seq_local_indices = []

    return sequences


def sequences_to_dataframe(sequences: List[RuptureSequence]) -> pd.DataFrame:
    """
    Converts a list of :class:`RuptureSequence` objects into a tidy DataFrame.

    Args:
        sequences: List returned by :func:`find_rupture_sequences`.

    Returns:
        DataFrame with one row per sequence and columns:
        ``sequence_id``, ``n_events``, ``start_time_year``, ``end_time_year``,
        ``duration_years``, ``fault_coverage_fraction``, ``covered_distance_km``,
        ``total_fault_length_km``, ``event_indices``.
    """
    if not sequences:
        return pd.DataFrame(columns=[
            "sequence_id", "n_events", "start_time_year", "end_time_year",
            "duration_years", "fault_coverage_fraction",
            "covered_distance_km", "total_fault_length_km", "event_indices",
        ])

    rows = []
    for seq_id, seq in enumerate(sequences):
        rows.append({
            "sequence_id": seq_id,
            "n_events": len(seq.event_indices),
            "start_time_year": seq.start_time_year,
            "end_time_year": seq.end_time_year,
            "duration_years": seq.duration_years,
            "fault_coverage_fraction": seq.fault_coverage_fraction,
            "covered_distance_km": seq.covered_distance_m / 1000.0,
            "total_fault_length_km": seq.total_fault_length_m / 1000.0,
            "event_indices": seq.event_indices,
        })

    return pd.DataFrame(rows)


def get_sequence_catalog(
    sim_data: SimulationData,
    sequence: RuptureSequence,
) -> pd.DataFrame:
    """
    Returns the rows of ``sim_data.catalog`` corresponding to the events in a sequence.

    Args:
        sim_data: The standardized simulation data object.
        sequence: A single :class:`RuptureSequence` as returned by
            :func:`find_rupture_sequences`.

    Returns:
        A DataFrame subset of ``sim_data.catalog`` containing only the member events,
        with an added column ``sequence_event_order`` (0-based position within the sequence).
    """
    sub = sim_data.catalog.loc[sequence.event_indices].copy()
    sub["sequence_event_order"] = range(len(sub))
    return sub
