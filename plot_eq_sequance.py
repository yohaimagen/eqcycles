import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from dataclasses import dataclass
from pathlib import Path
import meshio
import geopandas as gpd
from scipy.spatial import cKDTree
import pygmt
import pyproj
from sklearn.linear_model import LinearRegression

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- Run Definitions ---
COMPARISON_RUNS = ['2', '4', '7']
MESH_PATHS      = ['./NAF_res_3.msh', './NAF_res_3.msh', './NAF.msh']
BASE_DIRS       = ['./output/', './output/', './output/']

COMPARISON_RUNS = ['7']
MESH_PATHS      = ['./NAF.msh']
BASE_DIRS       = ['./output/']

# COMPARISON_RUNS = ['7', '8']
# MESH_PATHS      = ['./NAF.msh', './NAF.msh']
# BASE_DIRS       = ['./output/', './output/']

# --- Reference Geometry ---
SHAPEFILE_PATH  = '../shapefiles/NAF_simplefied.shp'

# --- Plotting Thresholds ---
MW_LABEL_THRESHOLD = 7.5      # Only label earthquakes larger than this Mw
RUPTURE_THRESHOLD  = 0.05     # Slip (m) required to count as ruptured
RUPTURE_BIN_COUNT  = 500      # Number of bins along strike for rupture plotting

# --- Direction Analysis Config ---
PRE_EVENT_TIME    = 24 * 60 * 60   # seconds (Window before event)
POST_EVENT_TIME   = 24 * 60 * 60   # seconds (Window after event)
MIN_R2_THRESHOLD  = 0.3            # R-squared threshold to declare "clear direction"

# --- GMT Plot Settings ---
PANEL_WIDTH   = 8    # cm
PANEL_HEIGHT  = 15   # cm
PANEL_SPACING = 1.0  # cm
MAP_REGION    = [28, 42, 38, 42] # Lon/Lat for map view

# =============================================================================
# 2. DATA CLASSES & LOADERS
# =============================================================================

@dataclass
class Simulation_data:
    sr: np.ndarray    # Slip rate / Velocity
    state: np.ndarray # State variable (psi)
    tau: np.ndarray   # Shear stress
    sigma: np.ndarray # Normal stress
    time: np.ndarray  # Time vector
    coords: np.ndarray # Coordinates
    EQ_slip: np.ndarray  # Earthquake slip data
    cat: pd.DataFrame  # Earthquake catalog data
    verts: np.ndarray 
    limits: list

    @classmethod
    def load(cls, base_dir: str, mesh_path: str, run_id: str = '1') -> 'Simulation_data':
        """
        Loads simulation data from binary files in the specified directory.
        
        Args:
            base_dir: Path to the directory containing the .dat files.
            run_id: The identifier suffix for the files (e.g., '205').
        """
        
        mesh = meshio.read(mesh_path)
        triangles = mesh.cells_dict["triangle"]
        verts = mesh.points[triangles] 
        ncell = len(triangles)

        limits = [mesh.points[:,0].min(), mesh.points[:,0].max(),
                  mesh.points[:,1].min(), mesh.points[:,1].max(),
                  mesh.points[:,2].min(), mesh.points[:,2].max()]
        
        base_path = Path(base_dir)
        
        # 1. Determine ncell from the xyz file (assuming 3 columns: x, y, z)
        # We calculate ncell first because the reshape logic depends on it.
        xyz_file = base_path / f"xyz{run_id}.dat"
        if not xyz_file.exists():
            raise FileNotFoundError(f"Coordinate file not found: {xyz_file}")
            
        # Assuming xyz is also float64 binary. 
        # If xyz is text, use np.loadtxt(xyz_file) instead.
        coords = np.loadtxt(xyz_file)
        dep=coords[:,0]
        ncell=len(dep)
    
        # 2. Load the Time vector
        time_file = base_path / f"time{run_id}.dat"
        time_data = np.loadtxt(time_file) / (365*24*60*60)
        time_data = time_data[:, 1]

        # 3. Helper function to load, reshape, and transpose fields
        def load_field(filename):
            fpath = base_path / filename
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
            
            data_raw = np.fromfile(fpath, dtype=np.float64)
            
            # Reshape logic from your example:
            # reshape(-1, ncell) makes it (N_time, N_cell)
            # .T makes it (N_cell, N_time)
            data_reshaped = data_raw.reshape(-1, ncell).T
            
            # Sanity check: Ensure time dimension matches
            if data_reshaped.shape[1] != time_data.shape[0]:
                print(f"Warning: {filename} time steps ({data_reshaped.shape[1]}) "
                      f"do not match time vector ({time_data.shape[0]}).")
                
            return data_reshaped

        # 4. Load all fields
        sr_data = np.log10(np.abs(load_field(f"vel{run_id}.dat")))
        state_data = load_field(f"psi{run_id}.dat")
        tau_data = load_field(f"tau{run_id}.dat")
        sigma_data = load_field(f"sigma{run_id}.dat")
        
        EQ_slip = np.fromfile(base_path / f"EQslip{run_id}.dat", dtype=np.float64)
        EQ_slip = EQ_slip.reshape(-1, ncell).T
        cat = pd.read_csv(base_path / f"event{run_id}.dat", sep='\s+', header=None)
        cat.columns = ['Event_ID', 'Step', 'Time_sec', 'Mw', 'Hypo_Node'] + [f'Col_{i}' for i in range(5, cat.shape[1])]
        cat['Time_year'] = cat['Time_sec'] / (365*24*60*60)
        

        return cls(
            verts=verts,
            limits=limits,
            sr=sr_data,
            state=state_data,
            tau=tau_data,
            sigma=sigma_data,
            time=time_data,
            coords=coords,
            EQ_slip=EQ_slip,
            cat=cat
        )

def parse_simulation_log(log_path):
    params = {}
    patterns = {
        'b': r"Parameter b:\s+([\d\.]+)",
        'Dc': r"Dc:\s+([\d\.]+)",
        'A_VW': r"A_VW:\s+([\d\.]+)",
        'h_star': r"Nucleation Size \(h\*\) - 1:\s+([\d\.]+)"
    }
    if not os.path.exists(log_path):
        return "Log file not found", {}
    with open(log_path, 'r') as f:
        content = f.read()
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            params[key] = float(match.group(1)) if match else np.nan
    label = (f"a={params.get('A_VW')}, b={params.get('b')}, "
             f"Dc={params.get('Dc')}m\n"
             f"h*={params.get('h_star', 0)/1000:.1f}km")
    return label, params

# =============================================================================
# 3. ANALYSIS UTILS
# =============================================================================

def get_reference_trace(shapefile_path):
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    naf_gdf = gpd.read_file(shapefile_path)
    naf_cart = naf_gdf.to_crs(3857)
    line_geom = naf_cart.geometry.iloc[0]
    x_orig = np.array(line_geom.coords)[:, 0]
    y_orig = np.array(line_geom.coords)[:, 1]
    x_orig -= x_orig[0]
    y_orig -= y_orig[0]
    dists = np.sqrt(np.diff(x_orig)**2 + np.diff(y_orig)**2)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    return x_orig, y_orig, cum_dist, naf_gdf

def map_mesh_to_strike(sim_coords, ref_x, ref_y, ref_dist):
    total_len = ref_dist[-1]
    spacing = 1000.0 
    interp_dist = np.arange(0, total_len, spacing)
    trace_x = np.interp(interp_dist, ref_dist, ref_x)
    trace_y = np.interp(interp_dist, ref_dist, ref_y)
    trace_points = np.column_stack((trace_x * 1e-3, trace_y * 1e-3, np.zeros_like(trace_x)))
    tree = cKDTree(trace_points)
    distances, indices = tree.query(sim_coords[:, :3], k=1)
    node_along_strike = interp_dist[indices]
    return node_along_strike

def get_rupture_mask(sim, eq_idx, node_distances, k, threshold):
    max_dist = np.max(node_distances)
    bin_edges = np.linspace(0, max_dist, k + 1)
    bin_indices = np.digitize(node_distances, bin_edges)
    slip_data = sim.EQ_slip[:, eq_idx]
    is_ruptured = np.zeros(k, dtype=bool)
    for i in range(k):
        target_bin = i + 1
        mask = (bin_indices == target_bin)
        if np.any(mask):
            if np.max(slip_data[mask]) > threshold:
                is_ruptured[i] = True
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, is_ruptured

def analyze_rupture_direction(sim, event_idx, mesh_along_strike, verbose=False):
    """
    Analyzes the rupture direction for a specific event.
    """
    event_row = sim.cat.iloc[event_idx]
    event_time = event_row['Time_sec']
    
    if verbose:
        print(f"  Analyzing Dir: Event {event_idx} (Mw {event_row['Mw']:.1f})")

    # 1. Define Time Window
    t_start = event_time - PRE_EVENT_TIME
    t_end   = event_time + POST_EVENT_TIME
    
    # Convert sim.time (years) to seconds for searching
    t_sims_s = sim.time * (365*24*60*60)
    
    # Find indices in the time vector
    t_idx_start = np.argmin(np.abs(t_sims_s - t_start))
    t_idx_end   = np.argmin(np.abs(t_sims_s - t_end))
    
    if t_idx_end >= len(t_sims_s):
        t_idx_end = len(t_sims_s) - 1
        
    window_len = t_idx_end - t_idx_start
    if window_len < 2:
        return None

    # 2. Extract Data for Window
    sr_window = sim.sr[:, t_idx_start:t_idx_end] # This is log10(v)
    time_window = t_sims_s[t_idx_start:t_idx_end]
    
    # Convert log10 velocity back to linear for thresholding
    sr_linear = 10**sr_window
    
    # 3. Find Arrival Times
    is_rupturing = sr_linear > RUPTURE_THRESHOLD
    nodes_ruptured_indices = np.any(is_rupturing, axis=1)
    
    if not np.any(nodes_ruptured_indices):
        return None

    ruptured_node_ids = np.where(nodes_ruptured_indices)[0]
    ruptured_sr = is_rupturing[ruptured_node_ids, :]
    first_rup_indices = np.argmax(ruptured_sr, axis=1)
    
    arrival_times = time_window[first_rup_indices]
    node_dists = mesh_along_strike[ruptured_node_ids]
    
    # 4. Linear Regression
    X = node_dists.reshape(-1, 1)
    y = arrival_times
    
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    r2 = reg.score(X, y)
    
    direction_code = 0
    
    if r2 > MIN_R2_THRESHOLD:
        if slope > 0:
            direction_code = 1 # Positive Strike
        else:
            direction_code = -1 # Negative Strike
            
    if verbose:
        print(f"    Slope={slope:.2e}, R2={r2:.2f}, Code={direction_code}")
    
    return {'code': direction_code, 'slope': slope, 'r2': r2}

# =============================================================================
# 4. MAIN SCRIPT
# =============================================================================

def main():
    print("Loading reference fault geometry...")
    ref_x, ref_y, ref_dist, naf_gdf = get_reference_trace(SHAPEFILE_PATH)
    
    fig_rup = pygmt.Figure()
    fig_fault_trace = pygmt.Figure()
    
    pygmt.config(FORMAT_GEO_MAP='ddd', MAP_FRAME_TYPE='plain', FONT_TITLE="12p,Helvetica-Bold,black")
    
    fig_fault_trace.basemap(region=[-1200, 0, 0, 250], projection="X12c/2.5c", frame=['WNse', 'xa300a100', 'ya50a10'])
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    print("\nStarting Rupture History Plot...")
    
    for i, (run_id, mesh_path, base_dir) in enumerate(zip(COMPARISON_RUNS, MESH_PATHS, BASE_DIRS)):
        print(f"--> Processing Run {run_id} from {base_dir}")
        
        try:
            sim = Simulation_data.load(base_dir, mesh_path, run_id)
        except Exception as e:
            print(f"    Error loading run {run_id}: {e}")
            continue
            
        fig_fault_trace.plot(x=sim.coords[:,0], y=sim.coords[:,1], style="c0.05c", fill=colors[i % len(colors)], label=f"Run {run_id}")

        mesh_along_strike = map_mesh_to_strike(sim.coords, ref_x, ref_y, ref_dist)
        
        # Setup Panel
        x_shift = 2 if i == 0 else (PANEL_WIDTH + PANEL_SPACING)
        fig_rup.shift_origin(xshift=f"{x_shift}c")
        
        max_time = sim.time[-1]
        x_min = float(-np.max(mesh_along_strike) * 1e-3)
        x_max = 0.0
        y_min = 0.0
        y_max = float(max_time)
        
        region = [x_min, x_max, y_min, y_max]
        frame = ['WStr', 'xa300f100+lDistance [km]', 'ya500f100+lTime [years]'] if i == 0 else ['WStr', 'xa300f100+lDistance [km]', 'ya500f100']
        
        fig_rup.basemap(region=region, projection=f"X{PANEL_WIDTH}c/{PANEL_HEIGHT}c", frame=frame)
        
        if sim.cat.empty:
            continue
            
        for idx, row in sim.cat.iterrows():
            centers, is_rup = get_rupture_mask(sim, idx, mesh_along_strike, k=RUPTURE_BIN_COUNT, threshold=RUPTURE_THRESHOLD)
            
            if not np.any(is_rup):
                continue
                
            rup_locs = -centers[is_rup] * 1e-3 # Convert to km, negative for plot
            eq_time = row['Time_year']
            
            # Plot rupture bar
            fig_rup.plot(x=rup_locs, y=np.ones_like(rup_locs)*eq_time, 
                         style='c0.15c', fill='red', pen='0.1p,red')
            
            # Label Large Events & Rupture Direction
            if row['Mw'] > MW_LABEL_THRESHOLD:
                # 1. Text Label
                fig_rup.text(x=np.mean(rup_locs), y=eq_time + (max_time*0.025),
                             text=f"M{row['Mw']:.1f}", font="8p,Helvetica-Bold,black", justify="CM")
                
                # 2. Calculate Direction
                try:
                    res = analyze_rupture_direction(sim, idx, mesh_along_strike, verbose=True)
                    
                    if res and res['code'] != 0:
                        # Determine Arrow Geometry
                        # Slope > 0 (Positive Strike) means propagation 0 -> 1200km
                        # On plot, this is Left (-1200) -> Right (0). 
                        # So Angle = 0 deg.
                        
                        # Slope < 0 (Negative Strike) means propagation 1200 -> 0km
                        # On plot, this is Right (0) -> Left (-1200).
                        # So Angle = 180 deg.
                        
                        angle = 0 if res['code'] > 0 else 180
                        arrow_x = rup_locs.min() if res['code'] > 0 else rup_locs.max()
                        arrow_y = eq_time
                        
                        
                        fig_rup.plot(
                            data=[[arrow_x, arrow_y, angle, 1.0]], # Length 0.6 cm
                            style="v0.35c+e", # Head size determined by vector width, +e means arrow at end
                            pen="0.05c,black",
                            fill="black"
                        )
                except Exception as e:
                    print(f"Warning: Direction analysis failed for event {idx}: {e}")

    print("Showing Rupture Plot...")
    fig_rup.savefig("rupture_comparison.png")
    fig_fault_trace.legend(position="JTR+jTR+o0.2c", box="+gwhite+p1p")
    fig_fault_trace.savefig("fault_traces.png")

if __name__ == "__main__":
    main()