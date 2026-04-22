# Implementation Plan: Rupture Sequence Detection

## Goal

Identify **rupture sequences** in simulation catalogs — groups of earthquakes that together rupture the entire (or nearly entire) fault length within a bounded time window. This captures the concept that a full fault rupture can be achieved incrementally by a cascade of individual earthquakes.

**Example:** On a fault with segments A, B, C:
- `A → B → C` is a sequence.
- `B → A → (small B) → (small A) → C` is still a sequence because the combined coverage reaches the full fault.

---

## New File

**`src/eqcycles/analysis/sequences.py`**

This sits alongside the existing `rupture.py` and `scoring.py` in the `analysis/` sub-package and reuses their primitives.

---

## Data Structures

### `RuptureSequence` dataclass

```python
@dataclass
class RuptureSequence:
    event_indices: List[int]        # Catalog indices of all member events
    start_time_year: float          # Time of the earliest event in the sequence
    end_time_year: float            # Time of the last event that completed coverage
    duration_years: float           # end_time_year - start_time_year
    fault_coverage_fraction: float  # Fraction of fault bins covered (0.0–1.0)
    covered_distance_m: float       # Total fault length covered (meters)
    total_fault_length_m: float     # Total fault length used as reference (meters)
    segment_coverage: np.ndarray    # Boolean array (num_segments,) of which bins are covered
```

### `EventCoverage` (internal, as a DataFrame)

Precomputed per-event coverage table with columns:
- `event_idx`: catalog index
- `time_year`: `catalog['Time_year']`
- `Mw`: magnitude
- `min_dist_m`, `max_dist_m`: along-strike rupture extent (meters)
- `rupture_len_m`: `max_dist_m - min_dist_m`
- `segment_mask`: boolean array of which bins this event covers (stored as object column or separate array)

---

## Functions

### 1. `compute_event_coverage`

```python
def compute_event_coverage(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    slip_threshold: float = 0.05,
    num_segments: int = 100,
    min_mw: float = 6.5,
) -> Tuple[pd.DataFrame, np.ndarray]
```

**Purpose:** Pre-compute, for each qualifying event, which fault segments it covers.

**Implementation:**
1. Call `get_rupture_mask(sim_data, idx, mesh_along_strike, num_segments, slip_threshold)` for each event with `Mw >= min_mw` — this already exists in `rupture.py`.
2. Store the `(bin_centers, is_ruptured)` result as a row in the output.
3. Return:
   - A `pd.DataFrame` with one row per qualifying event (columns: `event_idx`, `time_year`, `Mw`, `min_dist_m`, `max_dist_m`, `rupture_len_m`)
   - A 2D `np.ndarray` of shape `(n_events, num_segments)` holding the boolean segment masks.

**Reuses:** `get_rupture_mask` from `rupture.py`.


### 2. `find_rupture_sequences`

```python
def find_rupture_sequences(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    slip_threshold: float = 0.05,
    num_segments: int = 100,
    coverage_threshold: float = 0.9,
    max_duration_years: float = 50.0,
    min_mw: float = 6.5,
    allow_overlapping: bool = False,
) -> List[RuptureSequence]
```

**Purpose:** Find all rupture sequences in the simulation catalog using a greedy forward-sweep.

**Parameters:**
- `coverage_threshold`: fraction of `num_segments` bins that must be covered (default 0.9 = 90% of fault length).
- `max_duration_years`: maximum allowed time between first and last event in a sequence.
- `allow_overlapping`: if `False` (default), after a sequence is found, the search restarts from the event *after* the sequence's last event. If `True`, the search restarts from the second event of the found sequence (exhaustive mode).

**Algorithm (greedy, non-overlapping):**

```
coverage_df, segment_masks = compute_event_coverage(...)
events sorted chronologically (catalog is already sorted)

sequences = []
i = 0
while i < n_events:
    accumulated = zeros(num_segments, bool)
    seq_indices = []
    for j in range(i, n_events):
        dt = events[j].time - events[i].time
        if dt > max_duration_years:
            break  # window exceeded, no sequence from i
        accumulated |= segment_masks[j]
        seq_indices.append(j)
        coverage_frac = accumulated.sum() / num_segments
        if coverage_frac >= coverage_threshold:
            # Found a sequence!
            record RuptureSequence(seq_indices, ...)
            i = j + 1  # advance past this sequence
            break
    else:
        i += 1  # no sequence starting at i, try i+1
return sequences
```

For `allow_overlapping=True`, replace `i = j + 1` with `i += 1`.


### 3. `sequences_to_dataframe`

```python
def sequences_to_dataframe(sequences: List[RuptureSequence]) -> pd.DataFrame
```

**Purpose:** Flatten the list of `RuptureSequence` objects into a tidy `pd.DataFrame` for inspection and export.

**Output columns:**
- `sequence_id`, `n_events`, `start_time_year`, `end_time_year`, `duration_years`, `fault_coverage_fraction`, `covered_distance_m`, `total_fault_length_m`, `event_indices` (as list)


### 4. `get_sequence_catalog` (convenience)

```python
def get_sequence_catalog(
    sim_data: SimulationData,
    sequence: RuptureSequence,
) -> pd.DataFrame
```

**Purpose:** Return the slice of `sim_data.catalog` containing only the events that belong to a given sequence. Useful as input for plotting functions.

---

## Parameters and Defaults

| Parameter             | Default | Rationale                                                   |
|-----------------------|---------|-------------------------------------------------------------|
| `num_segments`        | 100     | ~10 km bins on a 1000 km fault; fine enough to resolve gaps |
| `coverage_threshold`  | 0.9     | 90% coverage allows a small "wiggle room" as requested      |
| `max_duration_years`  | 50.0    | Reasonable inter-event gap for a cascading sequence         |
| `slip_threshold`      | 0.05 m  | Matches the existing default in `rupture.py`                |
| `min_mw`              | 6.5     | Filters out micro-seismicity                                |

All defaults should be configurable via function arguments.

---

## Integration with Existing Code

| Existing function         | Used in                      | How                                                 |
|---------------------------|------------------------------|-----------------------------------------------------|
| `get_rupture_mask`        | `compute_event_coverage`     | Directly called to get per-event bin coverage       |
| `project_to_fault_trace`  | Caller's responsibility      | User passes pre-computed `mesh_along_strike`        |
| `SimulationData.catalog`  | `find_rupture_sequences`     | Source of event times and magnitudes                |
| `rupture_sequence_matplotlib.py` | (future) visualization | Pass sequence event indices to existing plot functions |

**Caller pattern** (mirrors `scoring.py` usage):

```python
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.analysis.sequences import find_rupture_sequences, sequences_to_dataframe

mesh_along_strike = project_to_fault_trace(sim_data.coords, shapefile_path)
sequences = find_rupture_sequences(sim_data, mesh_along_strike, coverage_threshold=0.9)
df = sequences_to_dataframe(sequences)
```

---

## Edge Cases and Robustness

1. **No events above `min_mw`**: Return an empty list.
2. **Single event covers the full fault** (e.g., a mega-thrust): It becomes a 1-event sequence with `duration_years = 0`.
3. **Fault not fully coverable within `max_duration_years`**: Those start points simply yield no sequence.
4. **Duplicate/partial re-ruptures** (like the B→A→small-B→small-A→C example): Naturally handled — accumulated coverage is boolean OR, so re-rupturing an already-covered segment is ignored for coverage accounting but the event is still included in `event_indices`.
5. **Events not sorted by time**: Add an assertion / sort guard at the top of `find_rupture_sequences`.

---

## Testing

Add to `src/eqcycles/analysis/synthetic.py` (or a new test file) a helper:

```python
def generate_sequence_test_catalog(fault_length_km, segments, events_per_sequence, ...)
```

Validate that:
- A perfect A→B→C sequence is detected.
- A sequence with interleaved partial ruptures is also detected.
- Events exceeding `max_duration_years` are not grouped.
- `coverage_threshold=1.0` only finds sequences that cover all bins.

---

## File Summary

```
src/eqcycles/analysis/
    sequences.py        ← NEW: RuptureSequence dataclass + detection functions
```

No other files need modification. The new module is self-contained and composable with the existing visualization pipeline.
