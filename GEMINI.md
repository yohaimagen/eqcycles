# eqcycles Implementation Plan

This design prioritizes the "separation of concerns" you requested, ensuring that visualization logic is isolated, loaders are abstract, and the package remains extendable.

## 1. Directory Structure

The project follows a standard Python package structure with the main source code located in the `src` directory.

```
/
├── pyproject.toml          # Project metadata and build configuration
├── src/
│   └── eqcycles/           # The main Python package
│       ├── __init__.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── geometry.py
│       │   └── rupture.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── data.py
│       │   └── exceptions.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── hbi.py
│       └── vis/
│           ├── __init__.py
│           ├── rupture_sequence.py
│           ├── slip_rate_video.py
│           └── utils.py
├── scripts/                # Standalone scripts for execution
│   ├── plot_rupture.py
│   └── render_video.py
└── tests/                  # Unit and integration tests
    ├── __init__.py
    └── test_data_loader.py

```

## 2. Unit Testing

To ensure the reliability of the data loading and processing components, a dedicated test suite should be maintained in the `tests/` directory.

**Framework:** `pytest` is recommended for its simplicity and powerful features.

**`tests/test_data_loader.py`:**
*   **Purpose:** Verify that the `HBILoader` correctly loads data, handles different file formats, and returns a well-formed `SimulationData` object.
*   **Strategy:**
    1.  Create a fixture that generates a set of temporary, valid dummy data files (`xyz.dat`, `time.dat`, `vel.dat`, etc.) and a dummy mesh file.
    2.  Write a test function that uses this fixture to run the `HBILoader`.
    3.  Assert that the shape, data types, and values of the loaded numpy arrays and pandas DataFrames are correct.
    4.  Test edge cases, such as missing files or malformed data, and assert that the appropriate exceptions are raised.

Running tests can be done using the `pytest` command from the root directory.

## 3. Module Details & Migration Strategy

The following sections outline what code goes where.

### A. Core Data (core/data.py)

We separate the data structure from the file reading. This class is just a container.

**Class: `SimulationData`**

**Attributes:**
* Standardized arrays: `slip_rate`, `shear_stress`, `state_variable`, `time`, `coords`.
* `mesh`: Contains vertices, triangles (using `meshio` objects).
* `catalog`: A Pandas DataFrame for events (standardized column names).

**Methods:**
* `subset_time(t_start, t_end)`: Returns a new `SimulationData` object sliced in time.

### B. Input/Output (io/)

This handles the "Extension" requirement. You can add `pylith.py` later without breaking existing code.

**`io/base.py` (Abstract Interface):**
```python
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> SimulationData:
        """Must return a standardized SimulationData object."""
        pass
```

**`io/hbi.py` (Concrete Implementation):**
* **Logic:** Move the `load_field`, binary reshaping, and `meshio` reading logic here.
* **Arguments:** Accepts `base_dir` and `run_id`.
* **Output:** Returns `SimulationData`.

### C. Analysis (analysis/)

Isolating math from plotting allows you to "score" simulations later without generating images.

**`analysis/geometry.py`:**
* **Logic:** Move `get_reference_trace` and `map_mesh_to_strike` here.
* **Function:** `project_to_fault_trace(coords, shapefile_path)` -> returns 1D distance array.

**`analysis/rupture.py`:**
* **Logic:** Move `analyze_rupture_direction` and `get_rupture_mask` here.
* **New Feature:** Create a class `RuptureMetrics` that stores slope, R2, and direction code, separating the calculation from the print statement.

### D. Visualization (vis/)

This fulfills your requirement for "single .py for a single plot type".

**`vis/utils.py`:**
* **Content:**
    * `get_continuous_cmap()` (from your video script).
    * Standardized plot settings (font sizes, default DPI).
    * Helper to format time labels (Seconds -> Years).

**`vis/rupture_sequence.py`:**
* **Dependencies:** `pygmt`, `core.data`, `analysis.rupture`.
* **Class/Function:** `plot(sim_data, rupture_metrics, output_path, config=None)`.
* **Logic:** Pure PyGMT calls. It receives processed data (like rupture direction), it does not calculate it.

**`vis/slip_rate_video.py`:**
* **Dependencies:** `matplotlib`, `multiprocessing`.
* **Class:** `VideoRenderer`.
* **Logic:** Contains the `process_frame` function and the `ffmpeg` subprocess calls.

## 3. Development Workflow (Step-by-Step)

Here is how I recommend you build this:

**Step 1: The Skeleton** Create the folders and empty `__init__.py` files.

**Step 2: The Loader (The Foundation)** Move your `Simulation_data` class to `core/data.py`. Move the `load()` method logic to `io/hbi.py`. Test: Write a tiny script that loads data and prints `sim.slip_rate.shape`.

**Step 3: Geometry & Analysis** Move `map_mesh_to_strike` to `analysis/geometry.py`. Move `analyze_rupture_direction` to `analysis/rupture.py`. Refactor: Ensure these functions take `SimulationData` as input and return standard Python types (lists/dicts/DataFrames), not plot objects.

**Step 4: The Visualization Refactor** Create `vis/rupture_sequence.py`. Import your new loader and analysis modules. Copy the PyGMT logic. Replace the raw variable access with your new `SimulationData` attributes.

**Step 5: The Scripts** Create `scripts/plot_rupture.py`. It should look like this:

```python
from eqcycles.io.hbi import HBILoader
from eqcycles.vis.rupture_sequence import plot_rupture_sequence

loader = HBILoader(run_id='7')
sim_data = loader.load('./output/')
plot_rupture_sequence(sim_data, "rupture.png")
```

## 4. Future Extensibility Check

* **New Solver?** Create `io/pylith.py`.
* **New Plot?** Create `vis/stress_drop_map.py`.
* **New Analysis?** Create `analysis/scoring.py` (e.g., comparing `sim.catalog` vs `historical_catalog`).

