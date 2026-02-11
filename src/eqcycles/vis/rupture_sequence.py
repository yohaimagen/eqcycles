import pygmt
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
    "panel_spacing": 1.0,
    "map_region": [28, 42, 38, 42],
    "gmt_font_title": "12p,Helvetica-Bold,black",
}

def plot_rupture_sequence(
    sim_data: SimulationData,
    shapefile_path: str,
    output_path: str = None,
    config: Dict[str, Any] = None,
    fig: pygmt.Figure = None,
    add_rupture_direction: bool = False,
):
    """
    Generates a rupture sequence plot for a given simulation run using PyGMT.

    Args:
        sim_data (SimulationData): The loaded and standardized simulation data.
        shapefile_path (str): Path to the reference fault trace shapefile.
        output_path (str): The path to save the output PNG image.
        config (Dict[str, Any], optional): A dictionary to override default plotting settings.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    print(f"--> Processing Run for Rupture Sequence Plot...")
    
    # 1. Perform Geometric Analysis
    # Project the mesh coordinates to the 1D fault trace
    mesh_along_strike = project_to_fault_trace(sim_data.coords, shapefile_path)
    print(np.max(mesh_along_strike), np.min(mesh_along_strike))

    # 2. Setup PyGMT Figure
    if fig is None:
        fig = pygmt.Figure()
        pygmt.config(
            FORMAT_GEO_MAP='ddd', 
            MAP_FRAME_TYPE='plain', 
            FONT_TITLE=cfg["gmt_font_title"]
        )

    # Define plot region based on analysis results
    max_time = sim_data.time[-1]
    # Distances are in meters, convert to km for plotting
    x_min_km = 0
    x_max_km = np.max(mesh_along_strike)
    print(f"    Plotting region: x [{x_min_km:.1f}, {x_max_km:.1f}] km, time [0, {max_time:.2f}] years")    
    
    
    region = [x_min_km, x_max_km, 0.0, float(max_time)]
    
    # Define frame and labels
    frame = [
        'WStr', # West, South, top, right labels
        'xa300f100+l"Distance Along Strike [km]"',
        'ya500f100+l"Time [years]"'
    ]

    fig.basemap(
        region=region,
        projection=f"X{cfg['panel_width']}c/{cfg['panel_height']}c",
        frame=frame
    )

    if sim_data.catalog.empty:
        print("    Warning: No events in the catalog to plot.")
        fig.savefig(output_path)
        return

    # 3. Iterate Through Events and Plot
    for idx, event in sim_data.catalog.iterrows():
        if event['Mw'] < 6.5:
            continue
        # Get the rupture extent for this event
        centers, is_rup = get_rupture_mask(
            sim_data, 
            idx, 
            mesh_along_strike, 
            num_bins=cfg["rupture_bin_count"], 
            slip_threshold=cfg["rupture_threshold"]
        )
        
        if not np.any(is_rup):
            continue
            
        # Convert distances (meters) to km and make them negative for plotting direction
        rup_locs_km = centers[is_rup]
        eq_time_yr = event['Time_year']
        
        # Plot the horizontal bar representing the rupture extent
        fig.plot(
            x=[np.min(rup_locs_km), np.max(rup_locs_km)], 
            y=[eq_time_yr, eq_time_yr], 
            fill='red', 
            pen='0.15c,red'
        )
        
        # If the event is large enough, add a label and analyze its direction
        if event['Mw'] > cfg["mw_label_threshold"]:
            # Add Magnitude Label
            fig.text(
                x=np.mean(rup_locs_km), 
                y=eq_time_yr + (max_time * 0.025), # Offset label slightly above the line
                text=f"M{event['Mw']:.1f}", 
                font="8p,Helvetica-Bold,black", 
                justify="CM" # Center-Middle alignment
            )
            
            # Analyze rupture direction
            if add_rupture_direction:
                try:
                    metrics = analyze_rupture_direction(sim_data, idx, mesh_along_strike, verbose=True)
                    
                    if metrics and metrics.code != 0:
                        # Determine arrow direction based on propagation direction code
                        # code > 0: propagates towards increasing distance (right on plot)
                        # code < 0: propagates towards decreasing distance (left on plot)
                        angle = 0 if metrics.code > 0 else 180
                        arrow_x = rup_locs_km.min() if metrics.code > 0 else rup_locs_km.max()
                        
                        fig.plot(
                            data=[[arrow_x, eq_time_yr, angle, 0.6]], # x, y, angle, length (cm)
                            style="v0.3c+e",  # vector style, 0.3cm head, arrow at end
                            pen="1p,black",
                            fill="black"
                        )
                except Exception as e:
                    print(f"    Warning: Direction analysis failed for event {idx}: {e}")

    if output_path is not None:
        fig.savefig(output_path)

    return fig
