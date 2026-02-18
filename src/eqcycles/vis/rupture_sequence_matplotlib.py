import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from eqcycles.core.data import SimulationData
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.analysis.rupture import get_rupture_mask, analyze_rupture_direction

# Default configuration dictionary for plotting
DEFAULT_CONFIG = {
    "mw_label_threshold": 7.5,
    "rupture_threshold": 0.05,
    "rupture_bin_count": 500,
    "panel_width": 8,
    "panel_height": 15,
    "font_size_title": "12",
    "font_weight_title": "bold",
}

def plot_rupture_sequence_matplotlib(
    sim_data: SimulationData,
    shapefile_path: str,
    output_path: str = None,
    config: Dict[str, Any] = None,
    ax: plt.Axes = None,
    add_rupture_direction: bool = False,
    ploting_mag_treshold: float = 7.0,
    add_catalog_index: bool = False,
    verbose: bool = False
):
    """
    Generates a rupture sequence plot for a given simulation run using Matplotlib.

    Args:
        sim_data (SimulationData): The loaded and standardized simulation data.
        shapefile_path (str): Path to the reference fault trace shapefile.
        output_path (str): The path to save the output PNG image.
        config (Dict[str, Any], optional): A dictionary to override default plotting settings.
        ax (plt.Axes, optional): A matplotlib axes object to plot on. If None, a new figure and axes are created.
        add_rupture_direction (bool): If True, analyze and plot rupture direction arrows.
        ploting_mag_treshold (float): The minimum magnitude to plot.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    print(f"--> Processing Run for Rupture Sequence Plot (Matplotlib)...")
    
    # 1. Perform Geometric Analysis
    mesh_along_strike = project_to_fault_trace(sim_data.coords, shapefile_path)

    # 2. Setup Matplotlib Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(cfg['panel_width'], cfg['panel_height']))
    else:
        fig = ax.get_figure()

    # Define plot region based on analysis results, converting meters to negative km
    max_time = sim_data.time[-1]
    x_min_km = -np.max(mesh_along_strike) * 1e-3
    x_max_km = 0.0
    
    ax.set_xlim(x_min_km, x_max_km)
    ax.set_ylim(0.0, float(max_time) * 1.05) # Add 5% padding to the top of the y-axis
    
    ax.set_xlabel("Distance Along Strike [km]")
    ax.set_ylabel("Time [years]")
    ax.set_title(
        "Rupture Sequence", 
        fontsize=int(cfg["font_size_title"]), 
        fontweight=cfg["font_weight_title"]
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if sim_data.catalog.empty:
        print("    Warning: No events in the catalog to plot.")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax

    # 3. Iterate Through Events and Plot
    for idx, event in sim_data.catalog.iterrows():
        if event.Mw < ploting_mag_treshold:
            continue
        
        centers, is_rup = get_rupture_mask(
            sim_data, 
            idx, 
            mesh_along_strike, 
            num_bins=cfg["rupture_bin_count"], 
            slip_threshold=cfg["rupture_threshold"]
        )
        
        if not np.any(is_rup):
            continue
            
        rup_locs_km = -centers[is_rup] * 1e-3
        eq_time_yr = event['Time_year']
        
        ax.plot(
            rup_locs_km, 
            np.full_like(rup_locs_km, eq_time_yr), 
            'o',
            markersize=2,
            color='red', 
            markeredgecolor='red',
            markeredgewidth=0.1
        )
        
        if event['Mw'] > cfg["mw_label_threshold"]:
            ax.text(
                np.mean(rup_locs_km), 
                eq_time_yr + (max_time * 0.025),
                f"$M_w{event['Mw']:.1f}$", 
                fontsize=8, 
                fontweight="bold", 
                ha="center",
                va="center"
            )
            
            if add_catalog_index:
                ax.text(
                    np.mean(rup_locs_km) + 0.03 * (x_max_km - x_min_km), # Offset to the right
                    eq_time_yr + (max_time * 0.015), # Slightly below magnitude label
                    f"({idx})",
                    fontsize=7,
                    ha="left", # Align to the left of the offset
                    va="center",
                    color="gray"
                )
            
            if add_rupture_direction:
                try:
                    metrics = analyze_rupture_direction(sim_data, idx, mesh_along_strike, verbose=verbose)
                    
                    if metrics and metrics.code != 0:
                        arrow_length_data = 30
                        # print(f'arrow_length_data : {arrow_length_data}')
                        arrow_props = dict(
                            head_width=0.01 * max_time,
                            head_length=0.02 * (x_max_km - x_min_km),
                            fc='k',
                            ec='k',
                            lw=1
                        )
                        x_hypo = -metrics.hypocenter_dist * 1e-3
                        y_hypo = metrics.hypocenter_time / (365 * 24 * 60 * 60)
                        # Plot hypocenter star if data is available
                        if metrics.hypocenter_dist is not None and metrics.hypocenter_time is not None:
                            
                            ax.plot(x_hypo, y_hypo, '*', color='yellow', markersize=10, markeredgecolor='black', zorder=10)

                        # Unilateral, left-pointing arrow (prop. along increasing mesh_along_strike)
                        if metrics.code == 1:
                            dx = -arrow_length_data    # Negative dx = points LEFT
                            ax.arrow(x_hypo, y_hypo, dx, 0, **arrow_props)
                        
                        # Unilateral, right-pointing arrow (prop. along decreasing mesh_along_strike)
                        elif metrics.code == -1:
                            dx = arrow_length_data     # Positive dx = points RIGHT
                            ax.arrow(x_hypo, y_hypo, dx, 0, **arrow_props)

                        # Bilateral, two arrows from hypocenter
                        elif metrics.code == 2 and metrics.hypocenter_dist is not None:

                            # Arrow 1: left-pointing (for propagation along increasing mesh_along_strike)
                            ax.arrow(x_hypo, y_hypo, -arrow_length_data, 0, **arrow_props)
                            
                            # Arrow 2: right-pointing (for propagation along decreasing mesh_along_strike)
                            ax.arrow(x_hypo, y_hypo, arrow_length_data, 0, **arrow_props)

                except Exception as e:
                    print(f"    Warning: Direction analysis failed for event {idx}: {e}")

    if output_path is not None:
        print(f"--> Saving rupture plot to {output_path}")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax
