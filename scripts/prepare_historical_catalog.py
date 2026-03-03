import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import os

def main():
    ruptures_path = 'scripts/shapefiles/fault_rupture_20th_century.shp'
    trace_path = 'scripts/shapefiles/NAF_simplefied.shp'
    output_csv = 'historical_ruptures_20th_century.csv'
    output_plot = 'historical_rupture_sequence.png'

    if not os.path.exists(ruptures_path) or not os.path.exists(trace_path):
        print(f"Error: Shapefiles not found.")
        return

    print(f"Loading shapefiles...")
    trace_gdf = gpd.read_file(trace_path)
    # Ensure the trace is a single linestring
    if len(trace_gdf) > 1:
        trace_geom = trace_gdf.geometry.unary_union
    else:
        trace_geom = trace_gdf.geometry.iloc[0]
    
    # Project to EPSG:3857 for consistency with src/eqcycles/analysis/geometry.py
    trace_gdf_3857 = trace_gdf.to_crs(3857)
    line_geom_3857 = trace_gdf_3857.geometry.iloc[0]
    
    # Get reference coordinates for normalization
    x_orig = np.array(line_geom_3857.coords)[:, 0]
    y_orig = np.array(line_geom_3857.coords)[:, 1]
    x_offset, y_offset = x_orig[0], y_orig[0]
    
    # Create normalized line for projection
    x_norm, y_norm = x_orig - x_offset, y_orig - y_offset
    norm_line = LineString(np.column_stack((x_norm, y_norm)))
    
    ruptures_gdf = gpd.read_file(ruptures_path)
    # Project ruptures to same CRS
    ruptures_3857 = ruptures_gdf.to_crs(3857)
    
    results = []
    print(f"Projecting {len(ruptures_3857)} ruptures...")
    
    for idx, row in ruptures_3857.iterrows():
        rup_geom = row.geometry
        
        # Get start and end points in geographic coordinates for the CSV
        # (Using the original WGS84 coordinates)
        orig_rup_geom = ruptures_gdf.iloc[idx].geometry
        coords = list(orig_rup_geom.coords)
        lon_start, lat_start = coords[0]
        lon_end, lat_end = coords[-1]

        # Project all points of the rupture onto the fault trace to find extent
        rup_dists = []
        for pt in rup_geom.coords:
            pt_norm = (pt[0] - x_offset, pt[1] - y_offset)
            d = norm_line.project(Point(pt_norm))
            rup_dists.append(d)
        
        min_d = min(rup_dists)
        max_d = max(rup_dists)
        
        # The trace starts at the East (lon ~ 40) and goes West (lon ~ 29).
        # So dist=0 is the Easternmost point.
        # most_eastward_point = min_d
        
        results.append({
            'year': int(row.year),
            'lon_start': lon_start,
            'lat_start': lat_start,
            'lon_end': lon_end,
            'lat_end': lat_end,
            'most_eastward_m': min_d,
            'rupture_length_m': max_d - min_d,
            'location_m': min_d, # To fit into scoring.py style if needed
            'min_dist_m': min_d,
            'max_dist_m': max_d
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('year')
    df.to_csv(output_csv, index=False)
    print(f"Saved catalog to {output_csv}")

    # Plotting
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(8, 12))
    
    total_len_m = norm_line.length
    x_min_km = -total_len_m * 1e-3
    x_max_km = 0.0
    
    ax.set_xlim(x_min_km, x_max_km)
    
    # Mirroring the style of src/eqcycles/vis/rupture_sequence_matplotlib.py
    for _, row in df.iterrows():
        year = row['year']
        # Distance along strike in negative km
        d1 = -row['min_dist_m'] * 1e-3
        d2 = -row['max_dist_m'] * 1e-3
        
        # Plot as a thick line or a series of dots
        # To mimic the dots in rupture_sequence_matplotlib.py:
        dist_points = np.linspace(d1, d2, int(row['rupture_length_m']/1000) + 2)
        ax.plot(
            dist_points, 
            np.full_like(dist_points, year), 
            'o',
            markersize=2,
            color='red',
            markeredgecolor='red',
            markeredgewidth=0.1
        )
        
        # Add year label
        ax.text(
            np.mean([d1, d2]), 
            year + 2, # Small offset
            str(year),
            fontsize=9,
            ha='center',
            fontweight='bold'
        )

    ax.set_xlabel("Distance Along Strike [km] (0 = East)")
    ax.set_ylabel("Time [years]")
    ax.set_title("20th Century NAF Historical Rupture Sequence", fontweight='bold')
    
    # Adjust Y axis to have some padding
    years = df['year'].values
    ax.set_ylim(min(years) - 5, max(years) + 10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot to {output_plot}")

if __name__ == "__main__":
    main()
