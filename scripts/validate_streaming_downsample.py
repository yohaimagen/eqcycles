"""
Validation script for streaming downsampling.

Runs stream_spatially_downsample and stream_temporally_downsample over all
9 simulations in hbi_sims/output, then cross-checks a single simulation
against the original in-memory downsample_simulation to verify correctness.
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path so we can run without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eqcycles.io import HBILoader
from eqcycles.analysis.downsampling import (
    build_spatial_mapping,
    stream_spatially_downsample,
    stream_temporally_downsample,
    downsample_simulation,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path("/export/dump/ymagen/NAF/old_sims/hbi_sims_non_ovelaping")
SOURCE_MSH  = BASE_DIR / "NAF.msh"
TARGET_MSH  = BASE_DIR / "NAF_res5.msh"
OUTPUT_DIR  = BASE_DIR / "output"
SPATIAL_DIR = BASE_DIR / "downsampled_spatial"
TEMPORAL_DIR = BASE_DIR / "downsampled_temporal"

SIM_IDS   = [6,7]   # 1 … 9
K_TEMPORAL = 10
FIELDS     = ["slip_rate", "state_variable", "shear_stress", "normal_stress", "slip"]

VALIDATE_SIM  = 6   # which sim to cross-check against in-memory path
VALIDATE_FIELD = "slip_rate"   # field to compare


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ---------------------------------------------------------------------------
# Step 1: build mapping once (lightweight — no heavy fields)
# ---------------------------------------------------------------------------
banner("Building spatial mapping")
loader = HBILoader(str(SOURCE_MSH))
data_light = loader.load(str(OUTPUT_DIR), VALIDATE_SIM, load_heavy_fields=False)
t0 = time.time()
mapping = build_spatial_mapping(data_light, TARGET_MSH)
print(f"Mapping built in {time.time()-t0:.1f}s  |  "
      f"source={mapping.n_source} cells  →  target={mapping.n_target} cells")


# ---------------------------------------------------------------------------
# Step 2: stream all simulations
# ---------------------------------------------------------------------------
banner(f"Streaming {len(SIM_IDS)} simulations")
SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

for sim_id in SIM_IDS:
    t_sim = time.time()

    # Spatial
    spatial_marker = SPATIAL_DIR / f"xyz{sim_id}_spatial.dat"
    if not spatial_marker.exists():
        stream_spatially_downsample(
            source_dir=OUTPUT_DIR,
            run_id=sim_id,
            mapping=mapping,
            output_dir=SPATIAL_DIR,
            out_run_id=f"{sim_id}_spatial",
            fields=FIELDS,
            k_temporal=1,
        )
    else:
        print(f"  Sim {sim_id}: spatial already done, skipping.")

    # Temporal
    temporal_marker = TEMPORAL_DIR / f"xyz{sim_id}.dat"
    if not temporal_marker.exists():
        stream_temporally_downsample(
            source_dir=OUTPUT_DIR,
            run_id=sim_id,
            output_dir=TEMPORAL_DIR,
            out_run_id=sim_id,
            fields=FIELDS,
            k=K_TEMPORAL,
        )
    else:
        print(f"  Sim {sim_id}: temporal already done, skipping.")

    print(f"  Sim {sim_id}: done in {time.time()-t_sim:.1f}s")


# ---------------------------------------------------------------------------
# Step 3: correctness check — compare streaming vs in-memory for one sim
# ---------------------------------------------------------------------------
banner(f"Correctness check: sim={VALIDATE_SIM}, field={VALIDATE_FIELD}")

# In-memory reference (load one field only)
print("  Loading single field for in-memory reference...")
data_ref = loader.load(str(OUTPUT_DIR), VALIDATE_SIM, fields_to_load=[VALIDATE_FIELD])
t0 = time.time()
ref_ds = downsample_simulation(data_ref, TARGET_MSH)
print(f"  In-memory downsample done in {time.time()-t0:.1f}s")

ref_field = getattr(ref_ds, VALIDATE_FIELD)   # shape (n_target, ntime)

# Load streamed output via HBILoader
target_loader = HBILoader(str(TARGET_MSH))
streamed = target_loader.load(
    str(SPATIAL_DIR), f"{VALIDATE_SIM}_spatial",
    fields_to_load=[VALIDATE_FIELD],
)
streamed_field = getattr(streamed, VALIDATE_FIELD)

# Both fields are stored as log10(|vel|) after loading through HBILoader.
# The streaming path writes raw velocity (same as source), so loading back
# through HBILoader gives the same log10 transformation — shapes must match.
print(f"\n  ref shape    : {ref_field.shape}")
print(f"  streamed shape: {streamed_field.shape}")

if ref_field.shape != streamed_field.shape:
    print("  [FAIL] Shape mismatch!")
    sys.exit(1)

max_abs_diff = np.max(np.abs(ref_field - streamed_field))
mean_abs_diff = np.mean(np.abs(ref_field - streamed_field))
rel_err = max_abs_diff / (np.abs(ref_field).max() + 1e-30)

print(f"\n  max |diff|  : {max_abs_diff:.3e}")
print(f"  mean |diff| : {mean_abs_diff:.3e}")
print(f"  relative err: {rel_err:.3e}")

TOLERANCE = 1e-10
if max_abs_diff < TOLERANCE:
    print(f"\n  [PASS] Max diff {max_abs_diff:.2e} < {TOLERANCE:.0e} — streaming output matches in-memory reference.")
else:
    print(f"\n  [FAIL] Max diff {max_abs_diff:.2e} exceeds tolerance {TOLERANCE:.0e}.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: sanity-check file sizes for all streamed sims
# ---------------------------------------------------------------------------
banner("File size sanity check")
all_ok = True
for sim_id in SIM_IDS:
    for field in FIELDS:
        from eqcycles.analysis.downsampling import _FIELD_FILE_MAP, _read_field_info
        prefix = _FIELD_FILE_MAP[field]

        ncell_src, ntime = _read_field_info(OUTPUT_DIR, sim_id)

        # Spatial output: same ntime, n_target cells
        sp_path = SPATIAL_DIR / f"{prefix}{sim_id}_spatial.dat"
        expected_spatial = mapping.n_target * ntime * 8
        actual_spatial = sp_path.stat().st_size if sp_path.exists() else -1
        ok_sp = actual_spatial == expected_spatial

        # Temporal output: ceil(ntime/K) steps, ncell_src cells
        n_kept = len(range(0, ntime, K_TEMPORAL))
        tp_path = TEMPORAL_DIR / f"{prefix}{sim_id}.dat"
        expected_temporal = ncell_src * n_kept * 8
        actual_temporal = tp_path.stat().st_size if tp_path.exists() else -1
        ok_tp = actual_temporal == expected_temporal

        if not ok_sp or not ok_tp:
            print(f"  Sim {sim_id} / {field}:  spatial={'OK' if ok_sp else f'FAIL (got {actual_spatial}, exp {expected_spatial})'}  "
                  f"temporal={'OK' if ok_tp else f'FAIL (got {actual_temporal}, exp {expected_temporal})'}")
            all_ok = False

if all_ok:
    print(f"  All {len(SIM_IDS)*len(FIELDS)} field files have correct byte sizes.")

banner("Done")
