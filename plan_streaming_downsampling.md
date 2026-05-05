# Plan: Streaming (Memory-Efficient) Downsampling

## 1. Problem

The current `downsample_simulation` in `downsampling.py` requires the full
`SimulationData` object to already hold every heavy field in RAM before
downsampling begins.  For a large simulation with 5 fields × N_cells × N_time
`float64` values this quickly exhausts available memory:

| Field        | HBI file          | Stored shape in RAM      |
|--------------|-------------------|--------------------------|
| slip_rate    | `vel{id}.dat`     | `(N_cell, N_time) float64` |
| state_variable | `psi{id}.dat`   | same                     |
| shear_stress | `tau{id}.dat`     | same                     |
| normal_stress | `sigma{id}.dat`  | same                     |
| slip         | `slip{id}.dat`    | same                     |

All five fields are loaded simultaneously by `_load_and_reshape_field`, then
handed to `downsample_simulation` — meaning **5 × source** arrays live in memory
at once, plus the nascent target arrays.

Temporal downsampling in the caller script has the same issue: it loads a field
fully, then slices it, only freeing memory after the slice is assigned back.

---

## 2. Key Insight: Binary File Layout

HBILoader stores each field as a raw `float64` binary file laid out as:

```
[ cell_0_t0, cell_1_t0, …, cell_N_t0,   ← timestep 0
  cell_0_t1, cell_1_t1, …, cell_N_t1,   ← timestep 1
  … ]                                    ← shape (N_time, N_cell), row-major
```

`_load_and_reshape_field` reads this as `np.fromfile` → `reshape(ntime, ncell).T`
to get `(N_cell, N_time)`.

Because the layout is contiguous per time step, `np.memmap` can open the file
with `shape=(ntime, ncell)` and we can lazily read one row (`mmap[t, :]`) at a
time **without loading the full file into RAM**.

---

## 3. Proposed Solution: Two New Streaming Entry Points

### 3a. Streaming Spatial Downsampling

Split `downsample_simulation` into two phases:

**Phase 1 — `build_spatial_mapping()`**  
Pure geometry: loads mesh centroids/areas (small), computes the
`(indices_flat, offsets, weights_flat)` arrays that encode the area-weighted
aggregation from source to target.  Returns a lightweight `SpatialMapping`
dataclass.  **No heavy fields touched.**

**Phase 2 — `stream_spatially_downsample()`**  
Uses the mapping to stream each field file independently:

```
for each field file:
    open output binary (write mode)
    open input binary as np.memmap(shape=(ntime, ncell_source))
    for t in range(ntime):
        source_row = mmap[t, :]          # one time step, (ncell_source,)
        target_row = _aggregate_1d_numba(source_row, ...)  # (ncell_target,)
        write target_row to output
    close files
```

Peak RAM: **one source row + one target row per field** — i.e., `2 × ncell × 8`
bytes, regardless of N_time.

### 3b. Streaming Temporal Downsampling

```
for each field file:
    open input binary as np.memmap(shape=(ntime, ncell))
    kept_timesteps = range(0, ntime, k)
    open output binary (write mode)
    for t in kept_timesteps:
        write mmap[t, :] to output
    close files
```

Also writes the subsampled `time{id}.dat` and copies `xyz`, `event`, `EQslip`
unchanged.

### 3c. Combined (Spatial + Temporal) in a Single Pass

To minimise I/O, both operations can be fused:

```
kept_timesteps = range(0, ntime, k_temporal)
for t in kept_timesteps:
    source_row = mmap[t, :]
    target_row = _aggregate_1d_numba(source_row, mapping)
    write target_row
```

This produces a spatially AND temporally downsampled file in one read pass.

---

## 4. New API (proposed additions to `downsampling.py`)

### `SpatialMapping` dataclass

```python
@dataclass
class SpatialMapping:
    indices_flat: np.ndarray   # (M,) int64
    offsets: np.ndarray        # (n_target+1,) int64
    weights_flat: np.ndarray   # (M,) float64
    n_source: int
    n_target: int
    target_centroids: np.ndarray   # (n_target, 3)
    target_mesh: meshio.Mesh
```

### `build_spatial_mapping(source_data, target_mesh, scale_factor, z_limit) -> SpatialMapping`

Extracts the geometry + mapping computation currently embedded in
`downsample_simulation`.  Takes a *lightweight* `SimulationData` (no heavy
fields required — only `coords` and `mesh` are used).

### `stream_spatially_downsample(source_dir, run_id, mapping, output_dir, out_run_id, fields, k_temporal=1)`

- Reads `ncell` from `xyz{run_id}.dat`, `ntime` from `time{run_id}.dat`.
- For each requested field, opens source file via `np.memmap`, streams
  time steps (strided by `k_temporal`), applies `_aggregate_1d_numba`, writes
  output binary.
- Handles the `log10`/`10**x` conversion for `slip_rate` (same as loader/saver).
- Copies `xyz`, `time`, `event`, `EQslip` using the existing lightweight mapping
  (catalog `Hypo_Node` remapping, node tag remapping).

### `stream_temporally_downsample(source_dir, run_id, output_dir, out_run_id, fields, k)`

- Opens each field via `np.memmap`, reads every `k`-th time step, writes output.
- Writes subsampled `time{out_run_id}.dat`.
- Copies `xyz`, `event`, `EQslip` unchanged (or subsamples EQslip events by
  catalog time filter if desired).

### Updated `downsample_simulation` (backward-compatible wrapper)

Keep the existing signature but internally delegate to `build_spatial_mapping`
+ `stream_spatially_downsample` so all callers keep working.

---

## 5. Changes to Existing Files

### `src/eqcycles/analysis/downsampling.py`

1. Extract geometry + mapping logic from `downsample_simulation` into
   `build_spatial_mapping`.
2. Add `SpatialMapping` dataclass.
3. Add `stream_spatially_downsample`.
4. Add `stream_temporally_downsample`.
5. Refactor `downsample_simulation` to call the new functions (no behaviour
   change, just delegation).

### `src/eqcycles/io/hbi.py`

No changes required.  The streaming functions read raw binary directly, bypassing
`HBILoader`.  `HBILoader` remains the correct entry point when the full in-memory
object is needed for analysis (e.g., OT scoring, visualisation).

### `src/eqcycles/core/data.py`

No changes required.  The streaming functions write binary + text files directly,
matching the format `HBILoader.load` expects.  `SimulationData.save` is still
used for lightweight metadata (coords, time, catalog).

---

## 6. Updated Script Integration

The `process_simulation` function in the caller script becomes:

```python
def process_simulation(sim_idx):
    # Load only lightweight metadata (no heavy fields)
    sim_loader = HBILoader(INPUT_MSH)
    data_light = sim_loader.load(OUTPUT_DIR, sim_idx, load_heavy_fields=False)

    # Build spatial mapping once (cheap — only uses mesh geometry)
    mapping = build_spatial_mapping(data_light, TARGET_MSH)

    # Branch A: spatial downsampling, full temporal resolution
    stream_spatially_downsample(
        source_dir=OUTPUT_DIR,
        run_id=sim_idx,
        mapping=mapping,
        output_dir=SPATIAL_SAVE_DIR,
        out_run_id=f'{sim_idx}_spatial',
        fields=FIELDS_TO_SLICE
    )

    # Branch B: temporal downsampling, full spatial resolution
    stream_temporally_downsample(
        source_dir=OUTPUT_DIR,
        run_id=sim_idx,
        output_dir=TEMPORAL_SAVE_DIR,
        out_run_id=sim_idx,
        fields=FIELDS_TO_SLICE,
        k=K_FACTOR
    )
```

Peak RAM per worker is now **O(2 × max(ncell_source, ncell_target))** instead of
**O(5 × ncell_source × ntime)**.

---

## 7. Implementation Steps

| # | Task | File |
|---|------|------|
| 1 | Add `SpatialMapping` dataclass | `downsampling.py` |
| 2 | Extract `build_spatial_mapping` from `downsample_simulation` | `downsampling.py` |
| 3 | Add `_read_field_info(source_dir, run_id)` helper to get `ncell`, `ntime` | `downsampling.py` |
| 4 | Implement `stream_spatially_downsample` using `np.memmap` | `downsampling.py` |
| 5 | Implement `stream_temporally_downsample` using `np.memmap` | `downsampling.py` |
| 6 | Refactor `downsample_simulation` to delegate to new functions | `downsampling.py` |
| 7 | Update caller script to use new streaming API | `scripts/` |
| 8 | Add unit tests for round-trip correctness vs. in-memory path | `tests/` |

---

## 8. Edge Cases and Considerations

- **`slip_rate` log10 conversion**: The loader applies `log10(|vel|)` on read and
  the saver applies `10**slip_rate` on write.  The streaming functions must
  replicate this for the `vel` file only.

- **`EQslip`**: Stored as `(num_events, ncell)` — not a time-series field.
  Spatial remapping is still event-by-event (same streaming pattern), temporal
  downsampling does not apply to it (it is already per-event, not per-timestep).

- **Empty neighbor sets**: The fallback `tree.query(k=1)` in the mapping
  computation is already handled in `build_spatial_mapping`; no change needed.

- **Catalog `Hypo_Node` remapping**: Must be re-run after spatial downsampling
  (already done in current code); the streaming function handles this as part of
  metadata writing.

- **`np.memmap` mode**: Use `mode='r'` for source files (read-only) and write
  output with `tofile` or a writable `memmap`. Using `tofile` in a loop avoids
  holding the full output array in memory.

- **Numba JIT warm-up**: `_aggregate_1d_numba` will JIT-compile on first call.
  Consider calling it once with a tiny dummy array at the start to avoid the
  compile delay mid-loop if that matters for the workflow.
