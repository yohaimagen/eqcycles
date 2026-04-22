# eqcycles Implementation Plan

This document outlines the structure and components of the `eqcycles` package, which is designed for the analysis and visualization of earthquake cycle simulation data. The design prioritizes separation of concerns, modularity, and extensibility.

## 1. Directory Structure

The project follows a standard Python package structure.

```
/
├── pyproject.toml          # Project metadata and build configuration
├── scripts/                # Standalone utility scripts
├── tests/                  # Unit tests
├── test_notebooks/         # Interactive test notebooks
└── src/
    └── eqcycles/           # The main Python package
        ├── __init__.py
        ├── core/
        │   └── data.py             # Core data structures (SimulationData)
        ├── io/
        │   ├── base.py             # Abstract data loader
        │   └── hbi.py              # Concrete loader for HBI data
        ├── analysis/
        │   ├── geometry.py         # Fault geometry projections
        │   ├── rupture.py          # Rupture propagation analysis
        │   ├── scoring.py          # Optimal Transport catalog scoring
        │   ├── synthetic.py        # Synthetic data generation
        │   ├── downsampling.py     # Numba-accelerated mesh downsampling
        │   ├── point_selection.py  # Mesh node lookup utilities
        │   ├── parameter_sweep.py  # 2D hyperparameter sweep over OT params
        │   └── sequences.py        # Rupture sequence detection
        └── vis/
            ├── utils.py                        # Visualization utilities
            ├── rupture_sequence.py             # Rupture plots with PyGMT
            ├── rupture_sequence_matplotlib.py  # Rupture plots with Matplotlib
            ├── slip_rate_video.py              # Slip rate video rendering
            ├── sweep_visualization.py          # Hyperparameter sweep plots
            ├── ot_visualization.py             # Optimal Transport coupling plots
            ├── diagnostics.py                  # Diagnostic time series / 3D snapshots
            └── slip_distribution.py            # Slip distribution visualization
```

## 2. Module Descriptions

The package is organized into four main sub-packages: `core`, `io`, `analysis`, and `vis`.

### Core (`src/eqcycles/core`)

*   **`data.py`**: Defines the `SimulationData` dataclass — the standardized container for all simulation outputs, including slip rate, stress, time series, mesh information, and the earthquake catalog. Provides `subset_time(t_start, t_end)` for time-range slicing and `save(output_dir, run_id)` for writing back to HBI-format `.dat` files.

### Input/Output (`src/eqcycles/io`)

*   **`base.py`**: Defines the `BaseLoader` abstract base class, ensuring all loaders expose a common `load(path, run_id) -> SimulationData` interface.
*   **`hbi.py`**: Concrete `HBILoader` for reading HBI (tandem-running code) binary output files. Handles XYZ mesh, time, EQslip, event catalog, and optional heavy fields (velocity, stress). Converts time from seconds to years and reshapes fields to match the mesh.

### Analysis (`src/eqcycles/analysis`)

*   **`geometry.py`**: Fault geometry utilities. `project_to_fault_trace` projects 3D mesh nodes onto a 1D fault trace (from a shapefile) using a KD-tree, returning along-strike distances in metres. `get_geometry_context` retrieves reference geometry and CRS.

*   **`rupture.py`**: Rupture characterization. `get_rupture_mask` returns a spatial boolean mask of ruptured bins. `get_rupture_locations_and_times` extracts per-node arrival times. `analyze_rupture_direction` performs linear regression to classify ruptures as unilateral-negative (−1), unclear (0), unilateral-positive (+1), or bilateral (2), returning a `RuptureMetrics` dataclass. `export_ruptures_to_geodataframe` exports ruptures as GeoDataFrame polylines.

*   **`scoring.py`**: Optimal Transport (OT) similarity scoring between earthquake catalogs. Uses the unbalanced Sinkhorn algorithm (POT library) with an optional **topological sequence penalty** (`seq_weight`) that penalises backward temporal jumps in the transport plan. Key functions: `prepare_event_data`, `prepare_sim_event_data`, `calculate_ot_score`, `get_transport_plan`, `evaluate_window_metrics`, `find_best_sequence` (sliding-window scan with joblib parallelism).

*   **`synthetic.py`**: Synthetic data generation for testing. `generate_linear_fault` creates a straight-line fault GeoDataFrame; `create_event_catalog` populates it with synthetic events from `(time, center_km, length_km)` tuples.

*   **`downsampling.py`**: Numba-JIT-accelerated mesh downsampling. `downsample_simulation` maps a high-resolution simulation onto a coarser target mesh using area-weighted aggregation, with a configurable z-depth cutoff to avoid mixing across depth boundaries. Uses `@njit(parallel=True)` helpers for 1D and 2D field aggregation.

*   **`point_selection.py`**: `find_node_at_point` locates the nearest mesh node to a target (along-strike, depth) coordinate using a KD-tree.

*   **`parameter_sweep.py`**: `run_2d_parameter_sweep` performs a grid search over OT hyperparameters (`reg_m`, `seq_weight`). The outer loop (hyperparameter pairs) is sequential; the inner sliding-window scan is parallelised via joblib. Returns a tidy DataFrame of `SweepResult` entries including best score, timing, mass recovery, and inversion magnitude.

*   **`sequences.py`**: Rupture sequence detection. `compute_event_coverage` pre-computes which fault segments each event covers (filterable by magnitude). `find_rupture_sequences` uses a greedy single-pass accumulator: events are appended chronologically until a coverage threshold is reached, at which point the sequence is recorded and the accumulator resets. Returns a list of `RuptureSequence` dataclasses; `sequences_to_dataframe` converts them to tidy format.

### Visualization (`src/eqcycles/vis`)

*   **`utils.py`**: Common helpers — `get_continuous_cmap`, `format_time_label`, `create_slip_rate_cmap`, and default plot settings constants.

*   **`rupture_sequence.py`**: Generates space-time rupture sequence diagrams using PyGMT (publication-quality cartographic output). Optional rupture direction annotations.

*   **`rupture_sequence_matplotlib.py`**: Equivalent rupture sequence plots using Matplotlib. Supports overlaying historical events, sequence band colour-coding, rupture direction arrows, and event index labels.

*   **`slip_rate_video.py`**: `VideoRenderer` renders MP4 videos of slip-rate evolution across the fault mesh. Uses multiprocessing for frame rendering and FFmpeg for video assembly.

*   **`sweep_visualization.py`**: Visualises hyperparameter sweep results. `plot_pareto_front` shows inversion magnitude vs. mass recovery coloured by `seq_weight`; `plot_parameter_heatmap` produces a multi-panel heatmap over the `reg_m` × `seq_weight` grid.

*   **`ot_visualization.py`**: Visualises OT couplings between two catalogs. `plot_ot_window` draws historical and simulation events as horizontal bars connected by coupling-strength-proportional lines. Supports shared colorbars across multiple panels.

*   **`diagnostics.py`**: `plot_point_timeseries` produces stacked time-series subplots for multiple parameters at a single mesh node. `plot_3d_snapshot` renders a 3D view of a parameter field at a given time slice with optional point markers.

*   **`slip_distribution.py`**: Slip distribution visualization (recently added).

## 3. Development Workflow

The modular design supports an iterative development workflow:

*   **Adding a New Data Source**: Create a new loader class in the `io` directory that inherits from `BaseLoader` and implements the `load` method.
*   **Adding a New Analysis**: Create a new file in the `analysis` directory. Functions should take a `SimulationData` object as input and return processed data (numbers, arrays, or DataFrames).
*   **Adding a New Plot**: Create a new file in the `vis` directory. Plotting functions should take a `SimulationData` object and any pre-processed analysis results as input.

This structure ensures that data loading, analysis, and visualization are decoupled, making the codebase easier to maintain and extend.
