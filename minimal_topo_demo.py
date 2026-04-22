# %%
# =============================================================================
# Minimal Topological Penalty Demo  —  2-panel
#
# Historical order:  H0 (East, t=10)  →  H1 (West, t=12)   [East FIRST]
#
# Sim 1 — INVERTED chronological order (West FIRST, East SECOND in sim time):
#   S0 (idx=0): West,  t=11  ← West comes first in simulation time
#   S1 (idx=1): East,  t=13  ← East comes second in simulation time
#   OT matches H0(East)→S1(East, idx=1) and H1(West)→S0(West, idx=0)
#   → expected indices [1, 0]: INVERSION  (simulation order is West→East = opposite of history)
#   → time errors tiny (1–3 yr)  → LOW base OT  (looks better without penalty)
#
# Sim 2 — CORRECT chronological order (East FIRST, West SECOND in sim time):
#   S0 (idx=0): East,  t=5   ← East comes first in simulation time
#   S1 (idx=1): West,  t=25  ← West comes second (20 yr gap, vs 2 yr in history)
#   OT matches H0(East)→S0(East, idx=0) and H1(West)→S1(West, idx=1)
#   → expected indices [0, 1]: no inversion
#   → West event is offset (H1 at t=12, S1 at t=25 → 13 yr gap after time-shift) → HIGHER base OT
#
# Result:
#   Without penalty → Sim 1 wins  (wrong: inverted sim looks better)
#   With    penalty → Sim 2 wins  (correct: penalty exposes the inversion)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import ot

from eqcycles.analysis.synthetic import generate_linear_fault, create_event_catalog
from eqcycles.analysis.scoring   import prepare_event_data, normalize_coords, get_transport_plan

# ── Fault ─────────────────────────────────────────────────────────────────────
_SHP = '/tmp/minimal_topo_demo_fault.shp'
fault = generate_linear_fault(29, 40, 39, 40)   # E-W fault, ~830 km
fault.to_file(_SHP)

# ── OT config ─────────────────────────────────────────────────────────────────
# Large scale_t compresses the time axis → spatial distances dominate OT cost
CFG        = dict(reg=0.05, reg_m=0.5, scale_x=100_000.0, scale_t=500.0, scale_mass=100_000.0)
SEQ_WEIGHT = 0.1

# ── Historical: East first (t=10), West second (t=12) — only 2 yr apart ───────
# The tiny time gap means OT sees almost purely spatial distances.
HIST = [
    (10, 650, 250),   # H0: East rupture,  t = 10 yr
    (12, 110, 220),   # H1: West rupture,  t = 12 yr  (2 yr after H0)
]
hist_df = create_event_catalog(fault, HIST)

# ── Sim 1 — INVERTED (West → East in simulation time) ─────────────────────────
# In simulation time: S0(West, t=11) comes first, S1(East, t=13) comes second.
# This is the OPPOSITE of history (East→West).
# OT spatially matches H0(East)→S1(East, idx=1) and H1(West)→S0(West, idx=0).
# Expected indices [1, 0]: INVERSION.  Time errors: 1 yr and 3 yr → low base OT.
SIM1 = [
    (11, 110, 220),   # S0 (idx=0): West,  t=11  [first in sim time]
    (13, 650, 250),   # S1 (idx=1): East,  t=13  [second in sim time]
]

# ── Sim 2 — CORRECT (East → West in simulation time, matching history) ─────────
# In simulation time: S0(East, t=5) comes first, S1(West, t=25) comes second.
# Same spatial order as history. Expected indices [0, 1]: no inversion.
# Relative spacing is 20 yr (vs 2 yr in history) → after time-shift the West
# event is temporally offset from H1, giving a non-zero base OT score.
SIM2 = [
    ( 5, 650, 250),   # S0 (idx=0): East,  t=5   [first in sim time]
    (25, 110, 220),   # S1 (idx=1): West,  t=25  [second in sim time, 20 yr gap]
]

# ── Score function ─────────────────────────────────────────────────────────────
def ot_scores(hist_df, sim_df, cfg, shp, seq_weight):
    hc, hm = prepare_event_data(hist_df, shp)
    sc, sm = prepare_event_data(sim_df,  shp)

    hc_n = normalize_coords(hc.copy(), cfg['scale_x'], cfg['scale_t'])
    hc_n[:, 1] -= hc_n[:, 1].min()
    hm_n = hm / cfg['scale_mass']

    sc_n = normalize_coords(sc.copy(), cfg['scale_x'], cfg['scale_t'])
    sc_n[:, 1] -= sc_n[:, 1].min()
    sm_n = sm / cfg['scale_mass']

    P    = get_transport_plan(hc_n, hm_n, sc_n, sm_n, cfg)
    M    = ot.dist(hc_n, sc_n, metric='euclidean')
    base = float((P * M).sum())

    sim_idx   = np.arange(P.shape[1], dtype=float)
    row_sums  = P.sum(axis=1)
    safe_rows = np.where(row_sums > 1e-12, row_sums, 1.0)
    exp_idx   = P @ sim_idx / safe_rows
    inv_mag   = float(np.maximum(0, -np.diff(exp_idx)).sum())
    penalty   = seq_weight * inv_mag

    return base, penalty, P, exp_idx, sim_df

res1 = ot_scores(hist_df, create_event_catalog(fault, SIM1), CFG, _SHP, SEQ_WEIGHT)
res2 = ot_scores(hist_df, create_event_catalog(fault, SIM2), CFG, _SHP, SEQ_WEIGHT)

# ── Print table ────────────────────────────────────────────────────────────────
print(f"\n{'':30s}  {'Base OT':>9}  {'Penalty':>9}  {'Total':>9}")
print("-" * 64)
for name, (base, pen, *_) in [("Sim 1 (inverted: W→E)", res1), ("Sim 2 (correct:  E→W)", res2)]:
    w = "  ← WINS" if (base + pen) == min(res1[0]+res1[1], res2[0]+res2[1]) else ""
    print(f"{name:30s}  {base:9.4f}  {pen:9.4f}  {base+pen:9.4f}{w}")
print()
print("Without penalty → Sim 1 wins (lower base OT, but inverted order).")
print("With    penalty → Sim 2 wins (penalty exposes the inversion).")

# ── Plot ───────────────────────────────────────────────────────────────────────
hist_centers_lon = (hist_df.lon_start + hist_df.lon_end) / 2
hist_times_yr    = hist_df.time.values

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for ax, label, (base, pen, P, exp_idx, sim_df) in zip(
        axes,
        ["Sim 1 — INVERTED order\n(West → East in simulation time)",
         "Sim 2 — CORRECT order\n(East → West in simulation time)"],
        [res1, res2]):

    sim_clon = (sim_df.lon_start + sim_df.lon_end) / 2
    sim_t    = sim_df.time.values

    # Simulation ruptures — gray thick bars
    for k, row in sim_df.iterrows():
        ax.plot([row.lon_start, row.lon_end], [row.time, row.time],
                color='gray', alpha=0.6, linewidth=6,
                solid_capstyle='butt', zorder=2,
                label='Simulation' if k == sim_df.index[0] else '')

    # Historical ruptures — black thin bars
    for k, row in hist_df.iterrows():
        ax.plot([row.lon_start, row.lon_end], [row.time, row.time],
                color='black', linewidth=2.5,
                solid_capstyle='butt', zorder=3,
                label='Historical' if k == hist_df.index[0] else '')

    # OT connection lines
    threshold = 1e-3 * P.max()
    segs, wts = [], []
    for i in range(len(hist_df)):
        for j in range(len(sim_df)):
            w = P[i, j]
            if w > threshold:
                segs.append([(hist_centers_lon.iloc[i], hist_times_yr[i]),
                              (sim_clon.iloc[j],         sim_t[j])])
                wts.append(w)

    if segs:
        wts  = np.array(wts)
        norm = mcolors.Normalize(vmin=wts.min(), vmax=wts.max())
        rgba = cm.cool(norm(wts))
        nw   = (wts - wts.min()) / max(wts.max() - wts.min(), 1e-9)
        rgba[:, 3] = 0.2 + 0.75 * nw
        lws        = 0.5 + 2.5  * nw
        lc = LineCollection(segs, colors=rgba, linewidths=lws, zorder=4)
        ax.add_collection(lc)
        sm_cb = plt.cm.ScalarMappable(cmap=cm.cool, norm=norm)
        sm_cb.set_array([])
        cbar = fig.colorbar(sm_cb, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label('Mass transported', fontsize=9)

    # Expected-index labels on historical bars
    for i, (lon, t, ei) in enumerate(zip(hist_centers_lon, hist_times_yr, exp_idx)):
        ax.annotate(f'H{i}  ⟨idx⟩={ei:.2f}',
                    xy=(lon, t), xytext=(6, 6), textcoords='offset points',
                    fontsize=8.5, color='black')

    # Shade inversion region (where expected sim index goes backward)
    steps = np.diff(exp_idx)
    for i, step in enumerate(steps):
        if step < 0:
            t_lo = min(hist_times_yr[i], hist_times_yr[i+1])
            t_hi = max(hist_times_yr[i], hist_times_yr[i+1])
            ax.axhspan(t_lo, t_hi, alpha=0.12, color='red', zorder=0)
            mid_t = (t_lo + t_hi) / 2
            ax.text(0.98, mid_t, f'⚠ inversion\nΔ⟨idx⟩={step:.2f}',
                    fontsize=8, color='firebrick', va='center', ha='right',
                    transform=ax.get_yaxis_transform())

    total  = base + pen
    winner = "✓ WINS" if total == min(res1[0]+res1[1], res2[0]+res2[1]) else "✗ LOSES"
    ax.set_title(
        f"{label}\n"
        f"Base OT = {base:.4f}   Penalty = {pen:.4f}   Total = {total:.4f}   {winner}",
        fontsize=9, pad=8
    )
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Time (years)", fontsize=10)
    ax.set_xlim(28.5, 40.5)
    ax.legend(loc='upper right', fontsize=9)

fig.suptitle(
    f"Topological Sequence Penalty  (seq_weight={SEQ_WEIGHT})\n"
    "History: East→West.   Sim 1 is West→East (inverted) but temporally closer — "
    "the penalty correctly prefers Sim 2.",
    fontsize=10, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.show()

# %%
