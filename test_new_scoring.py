import pandas as pd


from eqcycles.io import HBILoader
from eqcycles.analysis.geometry import project_to_fault_trace
from eqcycles.vis.rupture_sequence_matplotlib import plot_rupture_sequence_matplotlib
from eqcycles.analysis.scoring import prepare_event_data, find_best_sequence, normalize_coords, calculate_ot_score, prepare_sim_event_data


sim_loader = HBILoader('../varing_dc//NAF_res5.msh')
sim = sim_loader.load('../varing_dc/downsampled_output', 21, fields_to_load=['slip_rate'])

MSH_PATH = '../varing_dc/NAF_res5.msh' 
SIM_DATA_DIR = '../varing_dc/downsampled_output'
TRACE_SHAPEFILE = '../shapefiles/NAF_simplefied.shp'
HISTORICAL_CSV = '../varing_dc/historical_ruptures_20th_century.csv'

hist_df = pd.read_csv(HISTORICAL_CSV)
# prepare_event_data expects a 'time' column
hist_df['time'] = hist_df['year']

hist_cords = hist_df[['most_eastward_m', 'year']].values
hist_mass = hist_df['rupture_length_m'].values

res = prepare_sim_event_data(sim, '../shapefiles/NAF_simplefied.shp')
cords, mass = res

config = dict(reg=0.1, reg_m=10.0, scale_x=100_000.0, scale_t=100.0, scale_mass=10_000.0, step_years=3, alpha=0.5)
window_edg = 100
res = find_best_sequence(hist_cords, hist_mass, cords, mass, config, window_edg, parallel_jobs=1)