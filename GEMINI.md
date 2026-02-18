# eqcycles Implementation Plan

This document outlines the structure and components of the `eqcycles` package, which is designed for the analysis and visualization of earthquake cycle simulation data. The design prioritizes separation of concerns, modularity, and extensibility.

## 1. Directory Structure

The project follows a standard Python package structure.

```
/
├── pyproject.toml          # Project metadata and build configuration
├── src/
│   └── eqcycles/           # The main Python package
│       ├── __init__.py
│       ├── core/
│       │   ├── data.py         # Core data structures
│       ├── io/
│       │   ├── base.py         # Abstract data loader
│       │   └── hbi.py          # Concrete loader for HBI data
│       ├── analysis/
│       │   ├── geometry.py     # Fault geometry projections
│       │   ├── rupture.py      # Rupture propagation analysis
│       │   ├── scoring.py      # Catalog similarity scoring
│       │   └── synthetic.py    # Synthetic data generation
│       └── vis/
│           ├── rupture_sequence.py             # Rupture plots with PyGMT
│           ├── rupture_sequence_matplotlib.py  # Rupture plots with Matplotlib
│           ├── slip_rate_video.py              # Slip rate video rendering
│           └── utils.py                        # Visualization utilities
└── ...
```

## 2. Module Descriptions

The package is organized into four main sub-packages: `core`, `io`, `analysis`, and `vis`.

### Core (`src/eqcycles/core`)

*   **`data.py`**: This module defines the core data structure for the entire package, the `SimulationData` dataclass. It acts as a standardized container for all simulation outputs, including slip rate, stress, time series, mesh information, and the earthquake catalog. It also provides a method to subset the data by a time range.

### Input/Output (`src/eqcycles/io`)

This sub-package handles loading data from different simulation formats.

*   **`base.py`**: Defines the `BaseLoader` abstract base class. This ensures that any new loader for a different simulation type will adhere to a common `load` method that returns a standardized `SimulationData` object, making the package modular.
*   **`hbi.py`**: A concrete implementation of `BaseLoader` for reading data from HBI (tandem-running code) simulations. It handles HBI's specific binary file formats and packages the data into the standard `SimulationData` object.

### Analysis (`src/eqcycles/analysis`)

This sub-package contains modules for performing scientific analysis on the simulation data.

*   **`geometry.py`**: Contains functions for geometric analysis. Its primary function, `project_to_fault_trace`, takes 3D simulation coordinates and projects them onto a 1D fault trace defined by a shapefile, calculating the along-strike distance for each point.
*   **`rupture.py`**: Focuses on analyzing individual earthquake ruptures. It provides functions to determine the spatial extent of a rupture (`get_rupture_mask`) and to analyze its propagation direction (`analyze_rupture_direction`), including distinguishing between unilateral and bilateral ruptures.
*   **`scoring.py`**: Implements a sophisticated scoring mechanism to compare earthquake catalogs (e.g., simulation vs. historical). It uses Optimal Transport (OT) to measure the "distance" between two space-time distributions of events, providing a quantitative similarity score.
*   **`synthetic.py`**: To facilitate testing and validation, this module provides tools to generate synthetic data. It can create simple linear fault geometries and populate them with synthetic earthquake catalogs.

### Visualization (`src/eqcycles/vis`)

This sub-package is dedicated to creating plots and animations from the data.

*   **`rupture_sequence.py`**: Generates space-time plots of earthquake rupture sequences using the `pygmt` library.
*   **`rupture_sequence_matplotlib.py`**: An alternative implementation for plotting rupture sequences using the `matplotlib` library. It provides similar functionality, including plotting rupture direction arrows.
*   **`slip_rate_video.py`**: Renders MP4 videos of the slip rate evolution across the fault mesh over time using `matplotlib` and `ffmpeg`.
*   **`utils.py`**: A collection of helper functions and constants for visualization, such as custom colormap generators, time label formatters, and default plot settings.

## 3. Development Workflow

The modular design supports an iterative development workflow:

*   **Adding a New Data Source**: Create a new loader class in the `io` directory that inherits from `BaseLoader` and implements the `load` method.
*   **Adding a New Analysis**: Create a new file in the `analysis` directory. The functions should take a `SimulationData` object as input and return processed data (e.g., numbers, arrays, or pandas DataFrames).
*   **Adding a New Plot**: Create a new file in the `vis` directory. The plotting function should take a `SimulationData` object and any pre-processed analysis results as input.

This structure ensures that data loading, analysis, and visualization are decoupled, making the codebase easier to maintain and extend.
