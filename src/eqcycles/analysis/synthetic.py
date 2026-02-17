import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

def generate_linear_fault(
    start_lon: float, start_lat: float, end_lon: float, end_lat: float
) -> gpd.GeoDataFrame:
    """
    Creates a simple, straight-line fault as a GeoDataFrame.

    Args:
        start_lon, start_lat: Coordinates for the start of the fault line.
        end_lon, end_lat: Coordinates for the end of the fault line.

    Returns:
        A GeoDataFrame containing the single fault trace.
    """
    line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
    gdf = gpd.GeoDataFrame([1], geometry=[line], crs="EPSG:4326")
    gdf.columns = ['id', 'geometry']
    return gdf

def create_event_catalog(
    fault_gdf: gpd.GeoDataFrame, event_definitions: list
) -> pd.DataFrame:
    """
    Creates a synthetic earthquake catalog by mapping events to a fault trace.

    Args:
        fault_gdf (gpd.GeoDataFrame): GeoDataFrame containing the fault line geometry.
        event_definitions (list): A list of tuples defining events:
                                  `[(time, center_on_fault_km, length_km), ...]`.

    Returns:
        A pandas DataFrame with columns: `time`, `lon_start`, `lat_start`, 
        `lon_end`, `lat_end`.
    """
    if fault_gdf.empty:
        raise ValueError("Input fault GeoDataFrame is empty.")
    
    fault_line = fault_gdf.geometry.iloc[0]
    
    # Project to a meter-based CRS to work with distances
    fault_line_utm = fault_gdf.to_crs(3857).geometry.iloc[0]
    
    total_length_m = fault_line_utm.length
    
    events = []
    for time, center_km, length_km in event_definitions:
        center_m = center_km * 1000
        length_m = length_km * 1000

        # Define start and end points in 1D along the UTM-projected line
        start_dist_m = center_m - length_m / 2
        end_dist_m = center_m + length_m / 2
        
        # Clamp the rupture to the ends of the fault
        start_dist_m = np.clip(start_dist_m, 0, total_length_m)
        end_dist_m = np.clip(end_dist_m, 0, total_length_m)

        # Get the Point objects at these distances
        start_point_utm = fault_line_utm.interpolate(start_dist_m)
        end_point_utm = fault_line_utm.interpolate(end_dist_m)

        # Create a temporary GeoSeries to transform points back to geographic CRS (lon/lat)
        points_utm = gpd.GeoSeries([start_point_utm, end_point_utm], crs=3857)
        points_lonlat = points_utm.to_crs(fault_gdf.crs)

        events.append({
            "time": time,
            "lon_start": points_lonlat.iloc[0].x,
            "lat_start": points_lonlat.iloc[0].y,
            "lon_end": points_lonlat.iloc[1].x,
            "lat_end": points_lonlat.iloc[1].y,
        })
        
    return pd.DataFrame(events)


def generate_sanity_check_catalogs(
    fault_gdf: gpd.GeoDataFrame,
    noise_level: float = 1.0,
    perturbation_level: float = 0.0,
    sim_duration: float = 500.0,
    repeat_historical: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a pair of historical and simulated catalogs for validation.

    Args:
        fault_gdf: The fault on which to generate events.
        noise_level: Multiplier for the number of random noise events.
        perturbation_level: Multiplier for the noise added to the embedded
                             historical events (time, location, length).
        sim_duration: Total duration of the simulated catalog in years.
        repeat_historical: Number of times to repeat the historical sequence.

    Returns:
        A tuple containing: (historical_catalog_df, simulated_catalog_df)
    """
    fault_length_km = fault_gdf.to_crs(3857).length / 1000
    
    # 1. Define the "historical" ground truth sequence
    historical_events = [
        (1939, 800, 350),  # Time (year), center_km, length_km
        (1943, 500, 280),
        (1999, 150, 150),
    ]
    historical_catalog = create_event_catalog(fault_gdf, historical_events)
    hist_duration = historical_catalog['time'].max() - historical_catalog['time'].min()

    # 2. Define the "simulated" sequence
    simulated_events = []
    
    # Embed the historical sequence `repeat_historical` times
    if repeat_historical > 0:
        start_times = np.linspace(
            hist_duration, sim_duration - hist_duration, repeat_historical
        )
        for t_start in start_times:
            for t, center, length in historical_events:
                # Add perturbation to the embedded events if level > 0
                time_offset = np.random.normal(0, 2.0 * perturbation_level) # std-dev of 2 years
                center_offset = np.random.normal(0, 10.0 * perturbation_level) # std-dev of 10 km
                length_offset = np.random.normal(0, 10.0 * perturbation_level) # std-dev of 10 km
                
                # Ensure length doesn't become negative
                new_length = max(1, length + length_offset)

                simulated_events.append(
                    (
                        t_start + (t - historical_events[0][0]) + time_offset,
                        center + center_offset,
                        new_length,
                    )
                )

    # Add random "noise" events based on the noise_level
    num_noise_events = int(50 * noise_level)
    if num_noise_events > 0:
        noise_times = np.random.uniform(0, sim_duration, num_noise_events)
        noise_centers = np.random.uniform(0, fault_length_km, num_noise_events)
        noise_lengths = np.random.uniform(10, 50, num_noise_events)
        
        for i in range(num_noise_events):
            simulated_events.append((noise_times[i], noise_centers[i], noise_lengths[i]))

    if not simulated_events:
        return historical_catalog, pd.DataFrame(columns=historical_catalog.columns)

    simulated_catalog = create_event_catalog(fault_gdf, simulated_events)
    
    simulated_catalog = simulated_catalog.sort_values(by='time').reset_index(drop=True)

    return historical_catalog, simulated_catalog
