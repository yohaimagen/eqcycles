# Along-Strike Profile Refactor & Input/Output Comparison Plan

## Motivation

`slip_distribution.py` repeats the same binning/averaging logic in every function:
depth-filter nodes → compute bin edges → `np.digitize` → `np.add.at` aggregation →
build a `valid` mask. This should live in one place. Separately, the plotting
primitive (line + fill on an axes) is also duplicated. Extracting both lets us
build a generic input-vs-output comparison panel without re-implementing any
existing logic.

---

## Step 1 — New analysis helper: `compute_along_strike_profile`

**File:** `src/eqcycles/analysis/geometry.py`
(geometry already owns `project_to_fault_trace`; along-strike aggregation is a
natural extension)

### Signature

```python
def compute_along_strike_profile(
    mesh_along_strike: np.ndarray,   # (n_nodes,)  metres
    values: np.ndarray,              # (n_nodes,)  or  (n_nodes, N)
    coords: np.ndarray,              # (n_nodes, 3); z-column used for depth filter
    num_bins: int = 200,
    max_depth_km: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    bin_centers_km : (num_bins,)       along-strike bin centres (negative = west)
    profile        : (num_bins,)  or  (num_bins, N)   depth-averaged, bin-averaged
    valid          : (num_bins,) bool  True where at least one node fell in the bin
    """
```

### What it does
1. Depth filter: `shallow_mask = |coords[:, 2]| <= max_depth_km`
2. Build `bin_edges = np.linspace(0, mesh_along_strike.max(), num_bins + 1)`
3. `bin_centers_km = -(bin_edges[:-1] + bin_edges[1:]) / 2 * 1e-3`
4. `bin_idx = np.clip(np.digitize(...) - 1, 0, num_bins - 1)`
5. `np.add.at` aggregation (works for 1-D and 2-D `values` alike)
6. Divide by `count_per_bin` where `valid`; return `(bin_centers_km, profile, valid)`

---

## Step 2 — New vis primitive: `plot_along_strike_profile`

**File:** `src/eqcycles/vis/slip_distribution.py` (public — intended as a building
block for user-defined plots, not just internal use)

### Signature

```python
def plot_along_strike_profile(
    ax: plt.Axes,
    bin_centers_km: np.ndarray,  # (num_bins,)
    values: np.ndarray,          # (num_bins,)
    valid: np.ndarray,           # (num_bins,) bool
    color: str = "#2166ac",
    fill_alpha: float = 0.25,
    linewidth: float = 1.5,
    label: str | None = None,
    alpha: float = 1.0,
) -> None:
```

Draws `ax.plot` + `ax.fill_between` for `values[valid]` vs `bin_centers_km[valid]`.
No axis labels or titles — those stay in the calling function so each high-level
function can customise them. Being public lets users call `compute_along_strike_profile`
+ `plot_along_strike_profile` directly to assemble custom panels.

---

## Step 3 — Refactor `slip_distribution.py`

Each of the four existing public functions is refactored to:

1. **Resolve event indices / time window** (unchanged logic)
2. **Sum `eq_slip` across selected events** (unchanged)
3. **Call `compute_along_strike_profile`** instead of inline binning
4. **Call `plot_along_strike_profile`** instead of inline plotting
5. Keep all axis decoration (labels, title, spines, text annotation) in place

No behaviour change; only internal structure changes.

### Specific fixes bundled in

- `plot_slip_distribution_time_window` has a bug: `rate_color` is referenced but
  never defined. Fix: use `slip_color` consistently (the rate is the same colour,
  just on a second axis — the current design is correct, just the variable name is
  wrong).

---

## Step 4 — New function: `plot_along_strike_comparison`

**File:** `src/eqcycles/vis/slip_distribution.py`

Plots two along-strike profiles side-by-side on the **same axes** with twin
y-axes, intended for comparing any two scalar fields derived from `SimulationData`
(e.g. seismic vs. aseismic slip, coseismic slip vs. long-term slip from `sim_data.slip`).

### Signature

```python
def plot_along_strike_comparison(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    left_field: np.ndarray,          # (n_nodes,) or (n_nodes, N) — already extracted
    right_field: np.ndarray,         # (n_nodes,) or (n_nodes, N)
    left_label: str = "Left field",
    right_label: str = "Right field",
    left_ylabel: str | None = None,
    right_ylabel: str | None = None,
    ax: plt.Axes | None = None,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    left_color: str = "#2166ac",
    right_color: str = "#d6604d",
    fill_alpha: float = 0.2,
    linewidth: float = 1.5,
    title: str | None = None,
    output_path: str | None = None,
) -> tuple:
```

**Returns:** `(fig, ax_left, ax_right)`

### What it does
1. Calls `compute_along_strike_profile` for both fields.
2. If `right_field` has a different physical scale, plots it on `ax.twinx()`.
3. Calls `plot_along_strike_profile` for each profile.
4. Adds legend combining both handles.

### Intended use cases

| `left_field`                              | `right_field`                                |
|-------------------------------------------|----------------------------------------------|
| `eq_slip[:, indices].sum(axis=1)` (seismic cumulative slip) | `sim_data.slip[:, -1] - sim_data.slip[:, 0]` (total cumulative slip over same window) |
| seismic cumulative slip                   | aseismic = total − seismic                   |
| `eq_slip[:, indices].sum(axis=1)`         | `slip_rate.mean(axis=1)` (time-mean log-V)   |

The caller is responsible for computing `left_field` and `right_field` from
`sim_data` before calling the function; this keeps the vis layer thin.

---

## Step 5 — Grid comparison: `plot_input_output_comparison`

**File:** `src/eqcycles/vis/slip_distribution.py`

Wraps `plot_along_strike_comparison` in a `plot_slip_distributions_comparison`-
style grid — one panel per sequence — with the left profile showing coseismic slip
and the right profile showing a reference field (e.g. total accumulated slip over
the same time window, or time-mean slip rate).

### Signature

```python
def plot_input_output_comparison(
    sim_data: SimulationData,
    mesh_along_strike: np.ndarray,
    sequences: List[RuptureSequence],
    reference_field: np.ndarray,     # (n_nodes,) or (n_nodes, N) — same for all panels
    reference_label: str = "Reference",
    reference_ylabel: str | None = None,
    ncols: int = 2,
    num_bins: int = 200,
    max_depth_km: float = 14.0,
    output_path: str | None = None,
) -> tuple:
```

**Returns:** `(fig, axs_flat)`

---

## Implementation Order

1. `compute_along_strike_profile` in `geometry.py` (pure analysis, no vis dependency)
2. `plot_along_strike_profile` in `slip_distribution.py`
3. Refactor four existing functions to use steps 1 & 2 (tests should pass unchanged)
4. `plot_along_strike_comparison`
5. `plot_input_output_comparison`
