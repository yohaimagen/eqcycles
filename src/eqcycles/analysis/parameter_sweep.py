import numpy as np
import pandas as pd
import itertools
from typing import Dict, Any, List, TypedDict, Union, Optional
from joblib import Parallel, delayed

from eqcycles.analysis.scoring import (
    find_best_sequence, 
    evaluate_window_metrics, 
    normalize_coords,
    prepare_sim_event_data
)

class SweepResult(TypedDict):
    reg_m: float
    seq_weight: float
    best_time: float
    best_score: float
    mass_recovery_pct: float
    inversion_magnitude: float
    sim_index: int

def run_2d_parameter_sweep(
    hist_coords: np.ndarray,
    hist_masses: np.ndarray,
    sim_coords: np.ndarray,
    sim_masses: np.ndarray,
    base_config: Dict[str, Any],
    reg_m_grid: Union[List[float], np.ndarray],
    seq_weight_grid: Union[List[float], np.ndarray],
    window_edg: float = 50.0,
    parallel_jobs: int = 1,
    sim_index: int = -1
) -> pd.DataFrame:
    """
    Performs a 2D parameter sweep over reg_m and seq_weight.
    
    The outer loop is sequential to avoid thread explosion, while the inner 
    find_best_sequence call uses its internal Joblib parallelization.

    Args:
        hist_coords: (N, 2) array of (along-strike location, time) for historical events.
        hist_masses: (N,) array of masses for historical events.
        sim_coords: (M, 2) array of (along-strike location, time) for simulation events.
        sim_masses: (M,) array of masses for simulation events.
        base_config: Dictionary with base OT parameters (scale_x, scale_t, scale_mass, reg).
        reg_m_grid: List of reg_m values to sweep.
        seq_weight_grid: List of seq_weight values to sweep.
        window_edg: Padding around the historical duration in the simulation window.
        parallel_jobs: Number of parallel jobs for window scanning.

    Returns:
        A pandas DataFrame containing SweepResult entries.
    """
    # 1. Pre-normalize data once (shared across all reg_m/seq_weight combinations)
    hist_duration = hist_coords[:, 1].max() - hist_coords[:, 1].min()
    
    # Normalize historical coords
    relative_hist_coords = hist_coords.copy()
    relative_hist_coords[:, 1] -= hist_coords[:, 1].min()
    norm_hist_coords = normalize_coords(
        relative_hist_coords, base_config['scale_x'], base_config['scale_t']
    )
    
    # Normalize masses
    hist_masses_norm = hist_masses / base_config['scale_mass']
    sim_masses_norm = sim_masses / base_config['scale_mass']

    # 2. Sort simulation data for consistent window isolation later
    sort_idx = np.argsort(sim_coords[:, 1])
    sorted_sim_coords = sim_coords[sort_idx]
    sorted_sim_times = sorted_sim_coords[:, 1]
    sorted_sim_masses_norm = sim_masses_norm[sort_idx]

    results = []
    
    # 3. Iterate through hyperparameter combinations
    for reg_m, seq_weight in itertools.product(reg_m_grid, seq_weight_grid):
        print(f"Sweep evaluating: reg_m={reg_m}, seq_weight={seq_weight}")
        
        # Update config with current sweep parameters
        config = base_config.copy()
        config['reg_m'] = reg_m
        config['seq_weight'] = seq_weight
        
        # Find best sequence for this configuration
        scan_results = find_best_sequence(
            hist_coords, hist_masses, sim_coords, sim_masses, 
            config, window_edg=window_edg, parallel_jobs=parallel_jobs
        )
        
        if scan_results.empty or scan_results['score'].isnull().all():
            continue
            
        # Identify the winning window
        best_idx = scan_results['score'].idxmin()
        best_time = scan_results.loc[best_idx, 'time']
        best_score = scan_results.loc[best_idx, 'score']
        
        # 4. Isolate the winning window for detailed metrics
        idx_start = np.searchsorted(sorted_sim_times, best_time - window_edg, side='left')
        idx_end = np.searchsorted(sorted_sim_times, best_time + hist_duration + window_edg, side='right')
        
        window_coords = sorted_sim_coords[idx_start:idx_end]
        window_masses_subset = sorted_sim_masses_norm[idx_start:idx_end]
        
        if len(window_coords) == 0:
            continue

        # Normalize the isolated window
        relative_window_coords = window_coords.copy()
        relative_window_coords[:, 1] -= best_time
        norm_window_coords = normalize_coords(
            relative_window_coords, config['scale_x'], config['scale_t']
        )
        
        # 5. Extract detailed physical metrics
        metrics = evaluate_window_metrics(
            norm_hist_coords, hist_masses_norm,
            norm_window_coords, window_masses_subset,
            config
        )
        
        # Compile result
        res: SweepResult = {
            'reg_m': reg_m,
            'seq_weight': seq_weight,
            'best_time': float(best_time),
            'best_score': float(best_score),
            'mass_recovery_pct': metrics['mass_recovery_pct'],
            'inversion_magnitude': metrics['inversion_magnitude'],
            'sim_index': sim_index
        }
        results.append(res)
        
    return pd.DataFrame(results)


def generate_sweep_tasks(
    sims_dict: Dict[int, Any],
    reg_m_grid: Union[List[float], np.ndarray],
    seq_weight_grid: Union[List[float], np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Generates a flattened list of tasks for the ensemble parameter sweep.
    """
    tasks = []
    for sim_index, sim_data in sims_dict.items():
        for reg_m, seq_weight in itertools.product(reg_m_grid, seq_weight_grid):
            tasks.append({
                'sim_index': sim_index,
                'sim_data': sim_data,
                'reg_m': reg_m,
                'seq_weight': seq_weight
            })
    return tasks


def _execute_single_task(
    task: Dict[str, Any],
    norm_hist_coords: np.ndarray,
    hist_masses_norm: np.ndarray,
    hist_duration: float,
    base_config: Dict[str, Any],
    window_edg: float,
    trace_shapefile: str
) -> Optional[SweepResult]:
    """
    Executes a single (simulation, reg_m, seq_weight) sweep unit.
    """
    sim_index = task['sim_index']
    sim_data = task['sim_data']
    reg_m = task['reg_m']
    seq_weight = task['seq_weight']

    # 1. Extract and prepare simulation event data
    sim_coords, sim_masses = prepare_sim_event_data(sim_data, trace_shapefile)
    if sim_coords.size == 0:
        return None

    # 2. Update config
    config = base_config.copy()
    config['reg_m'] = reg_m
    config['seq_weight'] = seq_weight
    config['is_hist_normalized'] = True

    # 3. Find best sequence (forced sequential)
    scan_results = find_best_sequence(
        norm_hist_coords, hist_masses_norm, sim_coords, sim_masses,
        config, window_edg=window_edg, parallel_jobs=1
    )
    
    if scan_results.empty or scan_results['score'].isnull().all():
        return None
        
    # Identify the winning window
    best_idx = scan_results['score'].idxmin()
    best_time = scan_results.loc[best_idx, 'time']
    best_score = scan_results.loc[best_idx, 'score']
    
    # 4. Isolate the winning window for detailed metrics
    # Sort simulation data (find_best_sequence does this but doesn't return it)
    sort_idx = np.argsort(sim_coords[:, 1])
    sorted_sim_coords = sim_coords[sort_idx]
    sorted_sim_times = sorted_sim_coords[:, 1]
    sorted_sim_masses_norm = (sim_masses[sort_idx]) / config['scale_mass']

    idx_start = np.searchsorted(sorted_sim_times, best_time - window_edg, side='left')
    idx_end = np.searchsorted(sorted_sim_times, best_time + hist_duration + window_edg, side='right')
    
    window_coords = sorted_sim_coords[idx_start:idx_end]
    window_masses_subset = sorted_sim_masses_norm[idx_start:idx_end]
    
    if len(window_coords) == 0:
        return None

    # Normalize the isolated window
    relative_window_coords = window_coords.copy()
    relative_window_coords[:, 1] -= best_time
    norm_window_coords = normalize_coords(
        relative_window_coords, config['scale_x'], config['scale_t']
    )
    
    # 5. Extract detailed physical metrics
    metrics = evaluate_window_metrics(
        norm_hist_coords, hist_masses_norm,
        norm_window_coords, window_masses_subset,
        config
    )
    
    # Compile result
    res: SweepResult = {
        'reg_m': reg_m,
        'seq_weight': seq_weight,
        'best_time': float(best_time),
        'best_score': float(best_score),
        'mass_recovery_pct': metrics['mass_recovery_pct'],
        'inversion_magnitude': metrics['inversion_magnitude'],
        'sim_index': sim_index
    }
    return res


def run_flattened_ensemble_sweep(
    hist_coords: np.ndarray,
    hist_masses: np.ndarray,
    sims_dict: Dict[int, Any],
    trace_shapefile: str,
    base_config: Dict[str, Any],
    reg_m_grid: Union[List[float], np.ndarray],
    seq_weight_grid: Union[List[float], np.ndarray],
    window_edg: float = 50.0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Executes the ensemble parameter sweep using flattened top-level parallelization.

    Args:
        hist_coords: (N, 2) array of (along-strike location, time) for historical events.
        hist_masses: (N,) array of masses for historical events.
        sims_dict: Dictionary mapping sim_index to SimulationData objects.
        trace_shapefile: Path to the fault trace shapefile.
        base_config: Dictionary with base OT parameters.
        reg_m_grid: List of reg_m values to sweep.
        seq_weight_grid: List of seq_weight values to sweep.
        window_edg: Padding around the historical duration.
        n_jobs: Number of parallel cores to use for the flattened task list.

    Returns:
        A concatenated pandas DataFrame containing results for all tasks.
    """
    tasks = generate_sweep_tasks(sims_dict, reg_m_grid, seq_weight_grid)
    
    # Pre-calculate shared historical properties
    hist_duration = hist_coords[:, 1].max() - hist_coords[:, 1].min()
    
    # Pre-normalize historical data once
    relative_hist_coords = hist_coords.copy()
    relative_hist_coords[:, 1] -= hist_coords[:, 1].min()
    norm_hist_coords = normalize_coords(
        relative_hist_coords, base_config['scale_x'], base_config['scale_t']
    )
    hist_masses_norm = hist_masses / base_config['scale_mass']
    
    print(f"Executing {len(tasks)} tasks across {n_jobs} workers...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_execute_single_task)(
            task, norm_hist_coords, hist_masses_norm, hist_duration,
            base_config, window_edg, trace_shapefile
        ) for task in tasks
    )
    
    # Filter out None results (from empty simulations)
    valid_results = [r for r in results if r is not None]
    
    return pd.DataFrame(valid_results)


def run_ensemble_parameter_sweep(
    hist_coords: np.ndarray,
    hist_masses: np.ndarray,
    sims_dict: Dict[int, Any],
    trace_shapefile: str,
    base_config: Dict[str, Any],
    reg_m_grid: Union[List[float], np.ndarray],
    seq_weight_grid: Union[List[float], np.ndarray],
    window_edg: float = 50.0,
    parallel_jobs: int = 10
) -> pd.DataFrame:
    """
    Evaluates a 2D parameter sweep across an ensemble of simulations.

    Args:
        hist_coords: (N, 2) array of (along-strike location, time) for historical events.
        hist_masses: (N,) array of masses for historical events.
        sims_dict: Dictionary mapping sim_index to SimulationData objects.
        trace_shapefile: Path to the fault trace shapefile.
        base_config: Dictionary with base OT parameters.
        reg_m_grid: List of reg_m values to sweep.
        seq_weight_grid: List of seq_weight values to sweep.
        window_edg: Padding around the historical duration.
        parallel_jobs: Number of parallel jobs for window scanning.

    Returns:
        A concatenated pandas DataFrame containing results for all simulations.
    """
    all_sweep_dfs = []

    for sim_index, sim_data in sims_dict.items():
        print(f"Processing Simulation Ensemble Index: {sim_index}")
        
        # Extract simulation event data
        sim_coords, sim_masses = prepare_sim_event_data(sim_data, trace_shapefile)
        
        if sim_coords.size == 0:
            print(f"Warning: No events found for simulation {sim_index}. Skipping.")
            continue

        # Run 2D sweep for this specific simulation
        sweep_df = run_2d_parameter_sweep(
            hist_coords, hist_masses, sim_coords, sim_masses,
            base_config, reg_m_grid, seq_weight_grid,
            window_edg=window_edg, parallel_jobs=parallel_jobs,
            sim_index=sim_index
        )
        
        all_sweep_dfs.append(sweep_df)

    if not all_sweep_dfs:
        return pd.DataFrame()

    return pd.concat(all_sweep_dfs, ignore_index=True)


def aggregate_ensemble_metrics(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics from an ensemble of simulations.

    Args:
        ensemble_df: The concatenated DataFrame from run_ensemble_parameter_sweep.

    Returns:
        A grouped DataFrame containing ensemble statistics (mean, std, success_rate).
    """
    # 1. Calculate success flag for each row
    ensemble_df = ensemble_df.copy()
    ensemble_df['success'] = (
        (ensemble_df['mass_recovery_pct'] > 80.0) & 
        (ensemble_df['inversion_magnitude'] < 1.0)
    ).astype(float)

    # 2. Group by hyperparameters
    grouped = ensemble_df.groupby(['reg_m', 'seq_weight'])

    # 3. Aggregate metrics
    aggregated = grouped.agg({
        'mass_recovery_pct': ['mean', 'std'],
        'inversion_magnitude': ['mean', 'std'],
        'best_score': ['mean', 'std'],
        'success': 'mean'
    })

    # 4. Flatten columns and rename
    aggregated.columns = [
        f"{col}_{stat}" if stat != '' else col 
        for col, stat in aggregated.columns
    ]
    aggregated = aggregated.rename(columns={'success_mean': 'success_rate'})
    
    return aggregated.reset_index()


