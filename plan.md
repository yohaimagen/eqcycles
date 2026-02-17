# Plan: Optimal Transport Scoring for Earthquake Sequences (Revised)

This document outlines a revised plan to implement a scoring algorithm based on Unbalanced Optimal Transport (OT). The goal is to identify the best match for a known historical earthquake sequence within a longer simulation catalog.

This version incorporates two key changes:
1.  **Direct Rupture Geometry:** Instead of using magnitude (`Mw`) as a proxy for event "mass", the model will use the geometric start and end points of each rupture to calculate its length (mass) and center (location).
2.  **Synthetic Sanity Check:** The plan includes the creation and use of a synthetic dataset to provide a clear, verifiable "ground truth" for testing and validating the scoring algorithm.

## 1. Directory Structure

The project structure will be updated to support the new requirements:

-   **`src/eqcycles/analysis/scoring.py`**: Will contain the core OT scoring logic.
-   **`src/eqcycles/analysis/synthetic.py`**: (New) A module to generate synthetic fault traces and earthquake catalogs for validation.
-   **`scripts/score_sequence.py`**: The primary CLI for running the analysis on real data.
-   **`scripts/run_synthetic_test.py`**: (New) A dedicated script to execute the sanity check, demonstrating the method's validity on a known problem.
-   **`tests/test_scoring.py`**: Unit tests, which will now use the synthetic data generator.

## 2. Implementation Steps

### Step A: Dependency Management

-   **Action:** Ensure the `ot` library (`pot` on PyPI) is added to the project's dependencies.

### Step B: Synthetic Data Generation (`synthetic.py`)

This new module is the foundation for our sanity check. It will create a simplified, controllable environment.

-   **Action:** Create the `src/eqcycles/analysis/synthetic.py` file.
-   **Contents:**
    1.  **`generate_linear_fault(start_lon, start_lat, end_lon, end_lat)`**: Creates a simple, straight-line fault and returns it as a GeoDataFrame, which can be saved as a shapefile.
    2.  **`create_event_catalog(fault_trace_path, event_definitions)`**:
        -   **Input:** Path to the fault shapefile and a list of event definitions, e.g., `[(time, center_on_fault_km, length_km), ...]`.
        -   **Action:** For each definition, it calculates the start and end points along the 1D fault trace, finds their corresponding (lon, lat) coordinates, and builds a catalog.
        -   **Output:** A pandas DataFrame with the required columns: `time`, `lon_start`, `lat_start`, `lon_end`, `lat_end`.
    3.  **`generate_sanity_check_catalogs(fault_trace_path)`**:
        -   **Action:**
            -   Defines a "historical" sequence of a few large events (e.g., three M7+ events over 50 years).
            -   Defines a longer, 500-year "simulated" sequence. This sequence will include the exact historical sequence embedded at a known time (e.g., starting at year 250), plus numerous smaller "noise" events throughout.
            -   Uses `create_event_catalog` to generate both the historical and simulated catalogs.
        -   **Output:** Returns `(historical_catalog_df, simulated_catalog_df)`.

### Step C: Core Scoring Logic (`scoring.py`)

This module is the heart of the analysis and will contain the functions that prepare data and perform the sliding window Optimal Transport comparison.

-   **Action:** Create/update the `src/eqcycles/analysis/scoring.py` file.
-   **Contents:**

    1.  **`prepare_event_data(catalog_df, shapefile_path)`**:
        -   **Input:** A pandas DataFrame with columns `time`, `lon_start`, `lat_start`, `lon_end`, `lat_end`.
        -   **Action:**
            -   This function will perform a vectorized projection of all start and end coordinates. It will create two arrays of points, one for all start `(lon, lat)` and one for all end `(lon, lat)`.
            -   It will call `analysis.geometry.project_to_fault_trace` on both arrays to get 1D along-strike distances (`dist_start` and `dist_end`).
            -   Calculate the event's **location** as the most easterly point: `x = np.maximum(dist_start, dist_end)`. Using `np.maximum` ensures this is vectorized and efficient.
            -   Calculate the event's **mass** as its rupture length: `L = np.abs(dist_end - dist_start)`.
        -   **Output:** A tuple of NumPy arrays: `(coords, masses)`, where `coords` is an `(N, 2)` array of `(x, time)` and `masses` is an `(N,)` array of `L`.

    2.  **`normalize_coords(coords: np.ndarray, scale_x: float, scale_t: float)`**:
        -   A simple helper function that applies the normalization: `x' = x / scale_x`, `t' = t / scale_t`.

    3.  **`calculate_ot_score(coords1, masses1, coords2, masses2, config)`**:
        -   A wrapper function that takes the prepared data for two event sets (historical and simulation window) and a configuration dictionary containing `reg` and `reg_m`.
        -   It computes the cost matrix `C = ot.dist(coords1, coords2, metric='euclidean')`.
        -   It returns the unbalanced transport score: `score = ot.unbalanced.sinkhorn_unbalanced2(masses1, masses2, C, reg=config['reg'], reg_m=config['reg_m'])`.

    4.  **`find_best_sequence(sim_data, historical_catalog, shapefile_path, config)`**:
        -   This is the main workflow function that orchestrates the entire analysis.
        -   **Detailed Workflow:**
            1.  **Prepare Historical Data:**
                -   Call `prepare_event_data` on the `historical_catalog` to get `hist_coords` (location, time) and `hist_masses` (length).
                -   Calculate the time duration of the historical sequence: `hist_duration = hist_coords[:, 1].max() - hist_coords[:, 1].min()`.
            2.  **Prepare Simulation Data:**
                -   Call `prepare_event_data` on the full `sim_data.catalog` to get `sim_coords` and `sim_masses`.
            3.  **Initialize Sliding Window:**
                -   Define the simulation time range: `sim_start_time = sim_coords[:, 1].min()`, `sim_end_time = sim_coords[:, 1].max()`.
                -   Define the step size for the sliding window (e.g., `step_years = 1.0`).
                -   Create an array of window start times to evaluate: `window_starts = np.arange(sim_start_time, sim_end_time - hist_duration, step_years)`.
                -   Initialize an empty list to store results: `results = []`.
            4.  **Loop Through Windows:**
                -   For each `t_start` in `window_starts`:
                    -   `t_end = t_start + hist_duration`.
                    -   **Create Window Subset:** Efficiently select the simulation events inside the current time window `[t_start, t_end)`.
                    -   `window_mask = (sim_coords[:, 1] >= t_start) & (sim_coords[:, 1] < t_end)`.
                    -   `window_coords = sim_coords[window_mask]`
                    -   `window_masses = sim_masses[window_mask]`
                    -   If the window is empty, store a very high score (e.g., `np.inf`) and continue to the next window.
                    -   **Time Normalization (Crucial):** To make the comparison independent of absolute simulation time, shift the time for both historical and window events to start at zero.
                        -   `relative_hist_coords = hist_coords.copy(); relative_hist_coords[:, 1] -= hist_coords[:, 1].min()`.
                        -   `relative_window_coords = window_coords.copy(); relative_window_coords[:, 1] -= t_start`.
                    -   **Spatial & Temporal Normalization:** Apply the scaling factors to the relative coordinates for both sets using the `normalize_coords` function.
                    -   **Calculate Score:** Call `calculate_ot_score` with the fully prepared historical and window data.
                    -   **Store Result:** Append the tuple `(t_start, score)` to the `results` list.
            5.  **Finalize and Return:**
                -   Convert the `results` list into a pandas DataFrame with columns `['time', 'score']`.
                -   Find the row with the minimum score to identify the best fit.
                -   Return this DataFrame.

### Step D: Scripts for Execution

#### `run_synthetic_test.py` (New Sanity Check Script)

This is the most important new deliverable for validation.

-   **Action:** Create the `scripts/run_synthetic_test.py` file.
-   **Contents:**
    1.  Generate the simple fault trace and save it to a temporary shapefile.
    2.  Call `generate_sanity_check_catalogs` to get the synthetic historical and simulation data.
    3.  Define a `config` dictionary with the OT parameters (`scale_x`, `scale_t`, etc.).
    4.  Call `find_best_sequence` with the synthetic data.
    5.  Check if the time of the best-fit (minimum score) matches the known embedded time from step 2.
    6.  Print a clear report: "Synthetic test PASSED/FAILED. Known time: XXX, Found time: YYY, Score: ZZZ".
    7.  Optionally, generate a plot of the score over time to visualize the "dip" at the correct location.

#### `score_sequence.py` (Main CLI)

-   **Action:** Create or update the `scripts/score_sequence.py` file.
-   **Contents:** The script's `argparse` will be updated to accept paths to catalogs that conform to the new `(time, lon_start, lat_start, lon_end, lat_end)` format, removing any options related to `Mw` conversion.

This revised plan provides a more robust and verifiable path to implementing the desired scoring feature. The synthetic test case will build confidence that the algorithm is working as expected before applying it to complex, real-world simulation data.
