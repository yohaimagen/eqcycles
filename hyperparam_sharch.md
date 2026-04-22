```markdown
# Implementation Plan: Hyperparameter Sweep for Unbalanced OT Sequence Matching

**Objective:** Implement a hyperparameter sweep system for $reg\_m$ (mass conservation) and $seq\_weight$ (topological sequence penalty). The system must account for the fact that the optimal time window within the simulation shifts dynamically as hyperparameters change. Logic and visualization must be strictly decoupled.

---

## 1. Architectural Overview

Since the top-performing window shifts depending on the configuration, the sweep cannot evaluate hyperparameters on a fixed static window. The system must perform a full catalog scan (or a smart targeted scan) for *each* hyperparameter combination. 

**Target Modules:**
* **Logic:** `eqcycles.analysis.parameter_sweep` (New module)
* **Metrics:** `eqcycles.analysis.scoring` (Append helper functions)
* **Visualization:** `eqcycles.vis.sweep_visualization` (New module)

---

## 2. Step-by-Step Implementation Directives for LLM Agent

### Step 1: Establish Core Data Structures
Create a data class or defined dictionary structure to hold the results of a single hyperparameter evaluation. 
* **Agent Task:** Define a `SweepResult` TypedDict or dataclass containing:
    * `reg_m`: float
    * `seq_weight`: float
    * `best_time`: float (The optimal start time found by the sliding window)
    * `best_score`: float (The OT score at that time)
    * `mass_recovery_pct`: float (Percentage of historical mass matched)
    * `inversion_magnitude`: float (Quantification of topological sequence violations)

### Step 2: Implement Metrics Extraction Function
Currently, `calculate_ot_score` computes the penalty internally but only returns a float. We need a way to extract physical metrics from the winning window.
* **Agent Task:** Add a function `evaluate_window_metrics(hist_coords, hist_masses, sim_window_coords, sim_window_masses, config)` to `eqcycles.analysis.scoring`.
    * Must call `get_transport_plan` to retrieve the coupling matrix $P$.
    * Calculate **Mass Recovery**: $\sum P / \sum \text{hist\_masses}$.
    * Calculate **Inversion Magnitude**: Using the dot product of $P$ and simulation indices to find expected chronological order, then summing the negative temporal steps (backward jumps).
    * Return a dictionary of these specific metrics.

### Step 3: Implement the Sweep Controller (Logic Module)
Create the core sweeping logic that iterates over the parameter grid, utilizing the existing parallelized `find_best_sequence`.
* **Agent Task:** Create `run_2d_parameter_sweep(hist_coords, hist_masses, sim_coords, sim_masses, base_config, reg_m_grid, seq_weight_grid, window_edg)` in `eqcycles.analysis.parameter_sweep`.
    * Iterate through the Cartesian product of `reg_m_grid` and `seq_weight_grid`.
    * For each combination, update the `config` dictionary.
    * Execute `find_best_sequence(...)` to find the dynamic `best_time`.
    * Isolate the `sim_window_coords` and `sim_window_masses` corresponding to that `best_time` (using the same logic currently in the notebook: `best_time - window_edg` to `best_time + hist_duration + window_edg`).
    * Pass the isolated window to the `evaluate_window_metrics` function from Step 2.
    * Compile all results into a single Pandas DataFrame.

### Step 4: Implement Visualization Module (Strictly Separated)
Create pure functions that only accept the resulting DataFrame from Step 3. No OT calculations or configuration dictionaries should exist here.
* **Agent Task:** Create `eqcycles.vis.sweep_visualization`. Implement the following functions:
    * `plot_pareto_front(sweep_df: pd.DataFrame)`: A scatter plot of `inversion_magnitude` (X) vs. `mass_recovery_pct` (Y). Color the points by `seq_weight` (using a LogNorm scale) and size the points by `reg_m`.
    * `plot_parameter_heatmap(sweep_df: pd.DataFrame, metric: str)`: A 2D heatmap (e.g., using seaborn) with `reg_m` on one axis, `seq_weight` on the other, colored by a specified metric (e.g., `best_score`, `mass_recovery_pct`, or `inversion_magnitude`).

---

## 3. Computational Complexity & Optimization Notes (For the Agent)

* **Warning:** Running `find_best_sequence` (which is already parallelized via Joblib) inside a nested loop of hyperparameters can cause a thread explosion or take an exceptionally long time.
* **Optimization Directive:** The agent should ensure that the outer hyperparameter loop is executed sequentially, while the inner window-scanning (`find_best_sequence`) retains its Joblib parallelization. Alternatively, if the hyperparameter grid is large, suggest an optional "coarse-to-fine" grid search strategy.
* **Normalization:** Instruct the agent to pre-normalize the `hist_coords` and `hist_masses` *once* before entering the hyperparameter loop, as the spatial and temporal scaling factors do not change during the `reg_m` and `seq_weight` sweep.

```