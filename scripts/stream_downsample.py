"""
Parallel spatial and temporal downsampling using streaming I/O.

Peak RAM per worker: O(2 * max(n_source, n_target)) — independent of ntime.
"""
import os
import pandas as pd
import multiprocessing as mp

from eqcycles.io import HBILoader
from eqcycles.analysis.downsampling import build_spatial_mapping, stream_spatially_downsample, stream_temporally_downsample

# --- Configuration ---
BASE_DIR = './'
INPUT_MSH = os.path.join(BASE_DIR, 'NAF.msh')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TARGET_MSH = os.path.join(BASE_DIR, 'NAF_res5.msh')
SUMMARY_CSV = os.path.join(BASE_DIR, 'sweep_summary.csv')

SPATIAL_SAVE_DIR = os.path.join(BASE_DIR, 'downsampled_spatial')
TEMPORAL_SAVE_DIR = os.path.join(BASE_DIR, 'downsampled_temporal')

K_FACTOR = 20
FIELDS_TO_PROCESS = ['slip_rate', 'state_variable', 'shear_stress', 'normal_stress', 'slip']

# Build mapping once (shared across simulations — same source and target mesh)
_MAPPING = None

def _get_mapping():
    global _MAPPING
    if _MAPPING is None:
        loader = HBILoader(INPUT_MSH)
        # Load only lightweight metadata — no heavy fields needed for the mapping
        data_light = loader.load(OUTPUT_DIR, 0, load_heavy_fields=False)
        _MAPPING = build_spatial_mapping(data_light, TARGET_MSH)
    return _MAPPING


def process_simulation(sim_idx):
    prefix = f"[Sim {sim_idx:02d}]"
    spatial_marker = os.path.join(SPATIAL_SAVE_DIR, f"xyz{sim_idx}_spatial.dat")
    temporal_marker = os.path.join(TEMPORAL_SAVE_DIR, f"xyz{sim_idx}.dat")

    if os.path.exists(spatial_marker) and os.path.exists(temporal_marker):
        print(f"{prefix} [SKIP] Already downsampled.")
        return (sim_idx, True, "Skipped")

    print(f"{prefix} --- Starting ---")
    try:
        mapping = _get_mapping()

        # Branch A: spatial downsampling, full temporal resolution
        if not os.path.exists(spatial_marker):
            print(f"{prefix} Spatial downsampling...")
            stream_spatially_downsample(
                source_dir=OUTPUT_DIR,
                run_id=sim_idx,
                mapping=mapping,
                output_dir=SPATIAL_SAVE_DIR,
                out_run_id=f"{sim_idx}_spatial",
                fields=FIELDS_TO_PROCESS,
                k_temporal=1,
            )

        # Branch B: temporal downsampling, full spatial resolution
        if not os.path.exists(temporal_marker):
            print(f"{prefix} Temporal downsampling (k={K_FACTOR})...")
            stream_temporally_downsample(
                source_dir=OUTPUT_DIR,
                run_id=sim_idx,
                output_dir=TEMPORAL_SAVE_DIR,
                out_run_id=sim_idx,
                fields=FIELDS_TO_PROCESS,
                k=K_FACTOR,
            )

        print(f"{prefix} [SUCCESS]")
        return (sim_idx, True, "Success")

    except Exception as e:
        print(f"{prefix} [ERROR] {e}")
        return (sim_idx, False, str(e))


def main():
    for d in [SPATIAL_SAVE_DIR, TEMPORAL_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    try:
        df = pd.read_csv(SUMMARY_CSV)
        sim_nums = df['sim_num'].astype(int).tolist()
    except FileNotFoundError:
        print(f"[!] {SUMMARY_CSV} not found, using default range.")
        sim_nums = list(range(26))

    num_cores = 1
    print(f"[*] Processing {len(sim_nums)} simulations with {num_cores} worker(s)...")

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_simulation, sim_nums)

    successful = [r[0] for r in results if r[1]]
    failed = [r[0] for r in results if not r[1]]
    print(f"\nCompleted: {len(successful)} | Failed: {len(failed)}")
    if failed:
        print(f"Failed sims: {failed}")


if __name__ == "__main__":
    main()
