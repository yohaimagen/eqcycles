import numpy as np
import pandas as pd
import pytest
from eqcycles.analysis.parameter_sweep import (
    run_2d_parameter_sweep, 
    run_ensemble_parameter_sweep, 
    run_flattened_ensemble_sweep,
    aggregate_ensemble_metrics
)
from eqcycles.vis.sweep_visualization import (
    plot_pareto_front, 
    plot_parameter_heatmap,
    plot_ensemble_pareto_front,
    plot_ensemble_robustness_heatmap
)

def test_run_2d_parameter_sweep_basic():
    # Setup synthetic data
    # hist: 2 events
    hist_coords = np.array([[10.0, 0.0], [20.0, 1.0]])
    hist_masses = np.array([5.0, 5.0])
    
    # sim: 5 events, containing the match around t=100
    sim_coords = np.array([
        [0.0, 0.0],
        [10.5, 100.0],
        [20.5, 101.0],
        [50.0, 200.0],
        [60.0, 210.0]
    ])
    sim_masses = np.array([1.0, 4.8, 5.2, 1.0, 1.0])
    
    base_config = {
        'reg': 0.1,
        'scale_x': 1.0,
        'scale_t': 1.0,
        'scale_mass': 1.0,
        'step_years': 50.0  # Large step for fast test
    }
    
    reg_m_grid = [10.0, 100.0]
    seq_weight_grid = [0.0, 10.0]
    
    # Run sweep
    # Using 1 parallel job for the test to avoid overhead
    sweep_df = run_2d_parameter_sweep(
        hist_coords, hist_masses, sim_coords, sim_masses,
        base_config, reg_m_grid, seq_weight_grid,
        window_edg=10.0, parallel_jobs=1
    )
    
    assert isinstance(sweep_df, pd.DataFrame)
    assert not sweep_df.empty
    assert len(sweep_df) == 4 # 2x2 grid
    assert 'mass_recovery_pct' in sweep_df.columns
    assert 'inversion_magnitude' in sweep_df.columns
    assert 'best_time' in sweep_df.columns
    assert 'sim_index' in sweep_df.columns

    # Test visualization functions (don't show, just ensure they run)
    import matplotlib
    matplotlib.use('Agg') # Non-interactive backend
    
    ax1 = plot_pareto_front(sweep_df)
    assert ax1 is not None
    
    ax2 = plot_parameter_heatmap(sweep_df)
    assert ax2 is not None

def test_aggregate_ensemble_metrics():
    # Mock ensemble results
    mock_ensemble_df = pd.DataFrame([
        {'reg_m': 10.0, 'seq_weight': 0.0, 'mass_recovery_pct': 90.0, 'inversion_magnitude': 0.5, 'best_time': 100.0, 'best_score': 0.1, 'sim_index': 0},
        {'reg_m': 10.0, 'seq_weight': 0.0, 'mass_recovery_pct': 85.0, 'inversion_magnitude': 0.6, 'best_time': 105.0, 'best_score': 0.15, 'sim_index': 1},
        {'reg_m': 100.0, 'seq_weight': 10.0, 'mass_recovery_pct': 70.0, 'inversion_magnitude': 2.0, 'best_time': 110.0, 'best_score': 0.5, 'sim_index': 0},
        {'reg_m': 100.0, 'seq_weight': 10.0, 'mass_recovery_pct': 75.0, 'inversion_magnitude': 1.5, 'best_time': 115.0, 'best_score': 0.4, 'sim_index': 1},
    ])
    
    agg_df = aggregate_ensemble_metrics(mock_ensemble_df)
    
    assert isinstance(agg_df, pd.DataFrame)
    assert len(agg_df) == 2
    assert 'mass_recovery_pct_mean' in agg_df.columns
    assert 'success_rate' in agg_df.columns
    
    # Check success_rate for reg_m=10.0 (both should succeed)
    row_10 = agg_df[agg_df['reg_m'] == 10.0].iloc[0]
    assert row_10['success_rate'] == 1.0
    
    # Check success_rate for reg_m=100.0 (neither should succeed)
    row_100 = agg_df[agg_df['reg_m'] == 100.0].iloc[0]
    assert row_100['success_rate'] == 0.0
    
    # Test ensemble visualization
    import matplotlib
    matplotlib.use('Agg')
    
    ax1 = plot_ensemble_pareto_front(agg_df)
    assert ax1 is not None
    
    ax2 = plot_ensemble_robustness_heatmap(agg_df)
    assert ax2 is not None

@pytest.fixture
def mock_sim_data():
    from eqcycles.core.data import SimulationData
    # Minimal mock of SimulationData
    class MockSim:
        def __init__(self, t_offset):
            self.catalog = pd.DataFrame({
                'Time_year': [t_offset, t_offset + 1.0],
                'Mw': [7.5, 7.5]
            })
            self.coords = np.zeros((2, 3)) # Dummy coords
            
    return {0: MockSim(100.0), 1: MockSim(200.0)}

def test_run_flattened_ensemble_sweep(mock_sim_data, monkeypatch):
    # Setup synthetic data
    hist_coords = np.array([[10.0, 0.0], [20.0, 1.0]])
    hist_masses = np.array([5.0, 5.0])
    
    base_config = {
        'reg': 0.1,
        'scale_x': 1.0,
        'scale_t': 1.0,
        'scale_mass': 1.0,
        'step_years': 50.0
    }
    
    # Mock prepare_sim_event_data to avoid geometry projections
    def mock_prepare(sim_data, shapefile):
        t0 = sim_data.catalog['Time_year'].iloc[0]
        coords = np.array([[10.5, t0], [20.5, t0 + 1.0]])
        masses = np.array([4.8, 5.2])
        return coords, masses
        
    monkeypatch.setattr("eqcycles.analysis.parameter_sweep.prepare_sim_event_data", mock_prepare)
    
    reg_m_grid = [10.0]
    seq_weight_grid = [0.0]
    
    # Run flattened sweep
    ensemble_df = run_flattened_ensemble_sweep(
        hist_coords, hist_masses, mock_sim_data, 
        "dummy.shp", base_config, reg_m_grid, seq_weight_grid,
        window_edg=10.0, n_jobs=1
    )
    
    assert isinstance(ensemble_df, pd.DataFrame)
    assert len(ensemble_df) == 2 # 2 sims * 1 hyperparameter
    assert 'sim_index' in ensemble_df.columns
    assert set(ensemble_df['sim_index']) == {0, 1}
