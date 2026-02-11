import pytest

def test_core_imports():
    """Tests that core components can be imported from the top-level package."""
    try:
        from eqcycles.core import SimulationData
        from eqcycles.core import exceptions
    except ImportError as e:
        pytest.fail(f"Failed to import from eqcycles.core. Details: {e}")

def test_io_imports():
    """Tests that I/O components can be imported."""
    try:
        from eqcycles.io import BaseLoader, HBILoader
    except ImportError as e:
        pytest.fail(f"Failed to import from eqcycles.io. Details: {e}")

def test_analysis_imports():
    """Tests that analysis components can be imported."""
    try:
        from eqcycles.analysis import project_to_fault_trace
        from eqcycles.analysis import get_rupture_mask
        from eqcycles.analysis import analyze_rupture_direction
        from eqcycles.analysis import RuptureMetrics
    except ImportError as e:
        pytest.fail(f"Failed to import from eqcycles.analysis. Details: {e}")

def test_vis_imports():
    """Tests that the vis module is importable."""
    try:
        from eqcycles import vis
    except ImportError as e:
        pytest.fail(f"Failed to import eqcycles.vis. Details: {e}")
