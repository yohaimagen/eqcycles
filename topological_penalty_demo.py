# %% [markdown]
# # Topological Sequence Penalty — Synthetic Demonstration
# Run Section 8 independently for the 3-panel rupture-trace comparison.
#
# **Problem**: Standard Unbalanced OT minimizes total transport cost without
# respecting temporal ordering. In earthquake cycle matching we want the
# matching to honour the *arrow of time*: if historical event H_i comes before
# H_j, the simulation events matched to H_i should appear *earlier* in the
# simulation than those matched to H_j.
#
# When OT violates this ordering we call it a **topological inversion**.
# The sequence penalty adds a term proportional to the total backward-jump
# magnitude in the expected simulation-index sequence.
#
# **Key mechanism** (from `scoring.py`):
# ```
# expected_sim_indices[i] = sum_j P[i,j] * j  /  sum_j P[i,j]
# inversion_magnitude     = sum_i max(0, -(expected_sim_indices[i+1]
#                                          - expected_sim_indices[i]))
# score = base_OT_score + seq_weight * inversion_magnitude
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import ot

np.random.seed(0)

# =============================================================================
# SECTION 1: Why inversions occur — the "phase mismatch" scenario
# =============================================================================
# We work entirely in *normalized* space-time coordinates, so no shapefile is
# needed.  The numbers match what scoring.py sees after normalize_coords():
#
#   scale_x = 100 km  →  space in [0, 1]
#   scale_t = 500 yr  →  time compressed to tiny values
#
# Because the time axis is squashed, SPATIAL proximity dominates the OT cost,
# and events that share a spatial segment (east / west) tend to be matched
# together regardless of their relative timing.

# %%
# --- Historical: 3 events, East–West–East pattern over ~100 yr -----------------
# After normalization (÷500 yr), the historical time span is only 0.20 units.
hist_coords = np.array([
    [0.80, 0.00],   # H0 : East, very early
    [0.20, 0.10],   # H1 : West, mid
    [0.80, 0.20],   # H2 : East, late
])
hist_masses = np.ones(3)

# --- Simulation A (phase-MATCHED): West–East–West–East–… starting West --------
# The simulation opens with a WEST event, while the historical opens East.
# With compressed time the OT "sees" this mainly as spatial mismatch.
simA_coords = np.array([
    [0.20, 0.00],   # SA0 : West  idx=0
    [0.80, 0.10],   # SA1 : East  idx=1   ← spatially close to H0, H2
    [0.20, 0.20],   # SA2 : West  idx=2   ← spatially close to H1
    [0.80, 0.30],   # SA3 : East  idx=3
    [0.20, 0.40],   # SA4 : West  idx=4
    [0.80, 0.50],   # SA5 : East  idx=5
])
simA_masses = np.ones(6)

# --- Simulation B (phase-MATCHED): East–West–East–West–… starting East -------
# The simulation now opens on the East side, perfectly aligned with the
# historical record.
simB_coords = np.array([
    [0.80, 0.00],   # SB0 : East  idx=0   ← perfect match for H0
    [0.20, 0.10],   # SB1 : West  idx=1   ← perfect match for H1
    [0.80, 0.20],   # SB2 : East  idx=2   ← perfect match for H2
    [0.20, 0.30],   # SB3 : West  idx=3
    [0.80, 0.40],   # SB4 : East  idx=4
    [0.20, 0.50],   # SB5 : West  idx=5
])
simB_masses = np.ones(6)

# %%
# -------------------------------------------------------------------------
# SECTION 2: Compute OT transport plans for both simulations
# -------------------------------------------------------------------------
REG   = 0.05   # Sinkhorn entropic regularization (low → plan is more concentrated)
REG_M = 0.5    # Marginal relaxation (unbalanced OT)

MA = ot.dist(hist_coords, simA_coords, metric='euclidean')
MB = ot.dist(hist_coords, simB_coords, metric='euclidean')

PA = ot.unbalanced.sinkhorn_unbalanced(hist_masses, simA_masses, MA, reg=REG, reg_m=REG_M)
PB = ot.unbalanced.sinkhorn_unbalanced(hist_masses, simB_masses, MB, reg=REG, reg_m=REG_M)

base_A = float(np.sum(PA * MA))
base_B = float(np.sum(PB * MB))

print(f"Base OT score  — Simulation A (phase-inverted): {base_A:.4f}")
print(f"Base OT score  — Simulation B (phase-matched) : {base_B:.4f}")

# %%
# -------------------------------------------------------------------------
# SECTION 3: Compute expected simulation indices & inversion magnitude
# -------------------------------------------------------------------------

def expected_indices_and_penalty(P, seq_weight=1.0):
    """Return (expected_sim_indices, inversion_magnitude, penalty)."""
    sim_idx = np.arange(P.shape[1], dtype=float)
    row_sums = P.sum(axis=1)
    safe_row = np.where(row_sums > 1e-12, row_sums, 1.0)
    exp_idx = P @ sim_idx / safe_row
    steps = np.diff(exp_idx)
    inv = np.maximum(0, -steps)
    return exp_idx, float(np.sum(inv)), seq_weight * float(np.sum(inv))

SEQ_WEIGHT = 2.0

exp_A, inv_mag_A, penalty_A = expected_indices_and_penalty(PA, SEQ_WEIGHT)
exp_B, inv_mag_B, penalty_B = expected_indices_and_penalty(PB, SEQ_WEIGHT)

print("\n=== Simulation A (phase-inverted) ===")
for i, (ei, step) in enumerate(zip(exp_A, np.diff(np.insert(exp_A, 0, np.nan)))):
    flag = "  ← INVERSION" if (i > 0 and step < 0) else ""
    print(f"  H{i}  expected sim idx = {ei:.2f}{flag}")
print(f"  Total inversion magnitude : {inv_mag_A:.4f}")
print(f"  Sequence penalty (×{SEQ_WEIGHT})   : {penalty_A:.4f}")
print(f"  Total score (base+penalty) : {base_A + penalty_A:.4f}")

print("\n=== Simulation B (phase-matched) ===")
for i, (ei, step) in enumerate(zip(exp_B, np.diff(np.insert(exp_B, 0, np.nan)))):
    flag = "  ← INVERSION" if (i > 0 and step < 0) else ""
    print(f"  H{i}  expected sim idx = {ei:.2f}{flag}")
print(f"  Total inversion magnitude : {inv_mag_B:.4f}")
print(f"  Sequence penalty (×{SEQ_WEIGHT})   : {penalty_B:.4f}")
print(f"  Total score (base+penalty) : {base_B + penalty_B:.4f}")

# %%
# -------------------------------------------------------------------------
# SECTION 4: Big visualisation — side-by-side comparison
# -------------------------------------------------------------------------

LABEL_SIM_A = "Sim A  (phase-inverted, W–E–W–E…)"
LABEL_SIM_B = "Sim B  (phase-matched,  E–W–E–W…)"

COLOR_EAST = '#2166ac'   # blue
COLOR_WEST = '#d6604d'   # red-orange
COLOR_HIST = 'black'
CMAP_CONN  = 'viridis'


def space_color(space_val):
    return COLOR_EAST if space_val > 0.5 else COLOR_WEST


def draw_ot_panel(ax, hist_c, sim_c, P, exp_idx, title, t_offset=0.0,
                  show_inversions=True):
    """Draw space-time events + OT connections on a single axes."""
    # Simulation events (circles)
    for j, sc in enumerate(sim_c):
        col = space_color(sc[0])
        ax.scatter(sc[0], sc[1], s=120, color=col, zorder=4,
                   edgecolors='k', linewidths=0.8)
        ax.text(sc[0] + 0.03, sc[1], f'S{j}', fontsize=8, va='center',
                color='dimgray')

    # Historical events (squares, shifted to best-fit time t_offset)
    for i, hc in enumerate(hist_c):
        col = space_color(hc[0])
        t_vis = hc[1] + t_offset
        ax.scatter(hc[0], t_vis, s=160, color=col, marker='s', zorder=5,
                   edgecolors=COLOR_HIST, linewidths=1.5)
        ax.text(hc[0] - 0.07, t_vis, f'H{i}', fontsize=9, va='center',
                fontweight='bold', color=COLOR_HIST)

    # OT connections (LineCollection coloured by weight)
    threshold = 3e-3 * P.max()
    segs, weights = [], []
    for i, hc in enumerate(hist_c):
        for j, sc in enumerate(sim_c):
            w = P[i, j]
            if w > threshold:
                segs.append([(hc[0], hc[1] + t_offset), (sc[0], sc[1])])
                weights.append(w)

    if segs:
        weights = np.array(weights)
        norm_w = (weights - weights.min()) / max(weights.max() - weights.min(), 1e-9)
        rgba = plt.cm.get_cmap(CMAP_CONN)(norm_w)
        rgba[:, 3] = 0.2 + 0.75 * norm_w           # dynamic alpha
        lws = 0.5 + 2.5 * norm_w
        lc = LineCollection(segs, colors=rgba, linewidths=lws, zorder=3)
        ax.add_collection(lc)

    # Expected index arrows + inversion markers
    steps = np.diff(exp_idx)
    for i in range(len(hist_c)):
        hc = hist_c[i]
        t_vis = hc[1] + t_offset
        ax.annotate(f'⟨idx⟩={exp_idx[i]:.1f}',
                    xy=(hc[0], t_vis), xytext=(-72, 0),
                    textcoords='offset points', fontsize=7.5,
                    color='black', va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    if show_inversions:
        for i, step in enumerate(steps):
            if step < 0:
                mid_t = (hist_c[i][1] + hist_c[i + 1][1]) / 2 + t_offset
                ax.axhspan(hist_c[i][1] + t_offset, hist_c[i + 1][1] + t_offset,
                           alpha=0.12, color='red', zorder=0)
                ax.text(1.06, mid_t, f'⚠ inversion\n   Δ={step:.2f}',
                        fontsize=7.5, color='firebrick', va='center',
                        transform=ax.get_yaxis_transform())

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Along-strike (normalized)", fontsize=9)
    ax.set_xlim(-0.25, 1.25)
    ax.invert_yaxis()
    ax.set_ylabel("Time (normalized)", fontsize=9)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.35, lw=0.8)


fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.45)

ax_A  = fig.add_subplot(gs[0, 0])
ax_B  = fig.add_subplot(gs[0, 1], sharey=ax_A)
ax_Ai = fig.add_subplot(gs[1, 0])
ax_Bi = fig.add_subplot(gs[1, 1])

draw_ot_panel(ax_A, hist_coords, simA_coords, PA, exp_A,
              f"{LABEL_SIM_A}\nBase score = {base_A:.3f}  |  Penalty = {penalty_A:.3f}  →  Total = {base_A+penalty_A:.3f}")

draw_ot_panel(ax_B, hist_coords, simB_coords, PB, exp_B,
              f"{LABEL_SIM_B}\nBase score = {base_B:.3f}  |  Penalty = {penalty_B:.3f}  →  Total = {base_B+penalty_B:.3f}",
              show_inversions=False)

# --- Bottom: expected-index trajectories ------------------------------------
for ax, exp, base, pen, label, col in [
        (ax_Ai, exp_A, base_A, penalty_A, LABEL_SIM_A, '#d6604d'),
        (ax_Bi, exp_B, base_B, penalty_B, LABEL_SIM_B, '#2166ac')]:

    ax.plot(exp, np.arange(len(exp)), 'o-', color=col, lw=2, ms=8, zorder=4)
    ax.axvline(np.arange(len(exp)).mean(), color='gray', lw=0.5, ls=':')

    # shade inversions
    for i in range(len(exp) - 1):
        if exp[i + 1] < exp[i]:
            ax.axhspan(i, i + 1, alpha=0.15, color='red')
            ax.annotate(f'⚠ backward jump\n   Δ={exp[i+1]-exp[i]:.2f}',
                        xy=(exp[i + 1], i + 0.5), xytext=(15, 0),
                        textcoords='offset points', fontsize=8,
                        color='firebrick', va='center',
                        arrowprops=dict(arrowstyle='->', color='firebrick', lw=0.8))

    for i, (e, name) in enumerate(zip(exp, ['H0', 'H1', 'H2'])):
        ax.annotate(name, (e, i), textcoords='offset points',
                    xytext=(6, -4), fontsize=8)

    ax.set_yticks(range(len(exp)))
    ax.set_yticklabels([f'H{i}' for i in range(len(exp))], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Expected simulation index  ⟨idx⟩", fontsize=9)
    ax.set_title(f"Expected-index trajectory\n"
                 f"Base={base:.3f}  Penalty={pen:.3f}  Total={base+pen:.3f}",
                 fontsize=9)
    ax.set_xlim(-0.5, 5.5)
    ax.grid(axis='x', alpha=0.25)

# Legend patches
east_p = mpatches.Patch(color=COLOR_EAST, label='East-side event')
west_p = mpatches.Patch(color=COLOR_WEST, label='West-side event')
hist_p = mpatches.Patch(facecolor='white', edgecolor='black', label='Historical event (■)')
inv_p  = mpatches.Patch(color='red', alpha=0.2, label='Inversion region')
fig.legend(handles=[east_p, west_p, hist_p, inv_p],
           loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Topological Sequence Penalty — Why Spatial Phase Mismatch Creates Inversions\n"
    f"(reg={REG}, reg_m={REG_M}, seq_weight={SEQ_WEIGHT}  |  space scale >> time scale → space dominates OT cost)",
    fontsize=11, fontweight='bold', y=1.02
)
plt.show()

# %%
# -------------------------------------------------------------------------
# SECTION 5: Score landscape — how the penalty changes which window wins
# -------------------------------------------------------------------------
# Build a long simulation (800 yr) containing one "phase-matched" segment
# (at t=300) and one "phase-inverted" segment (at t=600).  We sweep windows
# and show how the penalty boosts the score of the inverted window.

SIM_DT  = 50.0          # yr between events in simulation
N_SIM   = 16            # total simulation events (800 yr)
HIST_DT = 50.0          # yr between historical events

sim_times  = np.arange(N_SIM) * SIM_DT
# Alternating East–West pattern (simulation starts East)
sim_space_raw = np.where(np.arange(N_SIM) % 2 == 0, 0.80, 0.20)
# Introduce a phase-flip between events 10 and 11 (at t=500→550)
# so events 10–15 run West–East instead of East–West:
sim_space_raw[10:] = 1.0 - sim_space_raw[10:]   # flip phase

# Normalize time (scale_t = 500 yr)
sim_coords_long = np.column_stack([sim_space_raw, sim_times / 500.0])
sim_masses_long  = np.ones(N_SIM)

# Historical: 3 events (East–West–East), 0–100 yr
hist_coords_raw = np.array([
    [0.80, 0.00 / 500.0],
    [0.20, 0.10 / 500.0],   # ÷ scale_t = 500
    [0.80, 0.20 / 500.0],
])
hist_masses_raw = np.ones(3)

HIST_DUR = hist_coords_raw[:, 1].max() - hist_coords_raw[:, 1].min()

# %%
# Sliding window scan
STEP    = 1.0 / 500.0   # 1 yr in normalised units
EDGE    = 100 / 500.0   # 100 yr padding

t_min = sim_coords_long[:, 1].min()
t_max = sim_coords_long[:, 1].max()
window_starts = np.arange(t_min, t_max - HIST_DUR + STEP / 2, STEP)

sim_times_sorted = sim_coords_long[:, 1]

scores_no_pen  = []
scores_pen     = []
inv_magnitudes = []

for t_start in window_starts:
    t_end = t_start + HIST_DUR
    i0 = np.searchsorted(sim_times_sorted, t_start - EDGE, side='left')
    i1 = np.searchsorted(sim_times_sorted, t_end   + EDGE, side='right')

    wc = sim_coords_long[i0:i1].copy()
    wm = sim_masses_long[i0:i1].copy()

    if len(wc) == 0:
        scores_no_pen.append(np.inf)
        scores_pen.append(np.inf)
        inv_magnitudes.append(0.0)
        continue

    # Shift window time to start at 0 (matching what find_best_sequence does)
    wc[:, 1] -= t_start

    M = ot.dist(hist_coords_raw, wc, metric='euclidean')
    P = ot.unbalanced.sinkhorn_unbalanced(hist_masses_raw, wm, M, reg=REG, reg_m=REG_M)

    base = float(np.sum(P * M))
    _, inv_mag, pen = expected_indices_and_penalty(P, SEQ_WEIGHT)

    scores_no_pen.append(base)
    scores_pen.append(base + pen)
    inv_magnitudes.append(inv_mag)

scores_no_pen  = np.array(scores_no_pen)
scores_pen     = np.array(scores_pen)
inv_magnitudes = np.array(inv_magnitudes)
t_years = window_starts * 500.0   # back to years for plotting

best_no_pen = t_years[np.nanargmin(scores_no_pen)]
best_pen    = t_years[np.nanargmin(scores_pen)]

print(f"\nSliding-window results:")
print(f"  Best window WITHOUT penalty : t = {best_no_pen:.0f} yr")
print(f"  Best window WITH    penalty : t = {best_pen:.0f} yr")
print(f"  Phase-matched segment starts at t ≈ 0 yr  (East–West–East)")
print(f"  Phase-inverted segment starts at t = 500 yr (West–East–West)")

# %%
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Top: simulation space-time diagram ------------------------------------
ax = axes2[0]
for j, (sc, st) in enumerate(zip(sim_space_raw, sim_times)):
    col = space_color(sc)
    ax.scatter(st, sc, s=100, color=col, zorder=4, edgecolors='k', lw=0.7)

# Shade the phase-inversion region
ax.axvspan(500, 800, alpha=0.08, color='red', label='Phase-inverted segment')
ax.axvspan(0, 500, alpha=0.08, color='green', label='Phase-matched segment')
ax.set_ylabel("Along-strike\n(normalized)", fontsize=9)
ax.set_title("Simulation catalog (long run, 800 yr)\n"
             "Green = phase-matched to history  |  Red = phase-inverted",
             fontsize=10)
ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=8, loc='upper right')
east_p2 = mpatches.Patch(color=COLOR_EAST, label='East-side event')
west_p2 = mpatches.Patch(color=COLOR_WEST, label='West-side event')
ax.legend(handles=[east_p2, west_p2], fontsize=8, loc='upper left')

# --- Middle: score landscape -----------------------------------------------
ax = axes2[1]
ax.plot(t_years, scores_no_pen, lw=1.5, color='steelblue',
        label='Score WITHOUT penalty')
ax.plot(t_years, scores_pen,    lw=1.5, color='firebrick',
        label=f'Score WITH penalty (seq_weight={SEQ_WEIGHT})')
ax.axvline(best_no_pen, color='steelblue', lw=1.5, ls='--',
           label=f'Best (no penalty) → t={best_no_pen:.0f} yr')
ax.axvline(best_pen,    color='firebrick', lw=1.5, ls='--',
           label=f'Best (with penalty) → t={best_pen:.0f} yr')
ax.set_ylabel("OT Score", fontsize=9)
ax.set_title("Sliding-window OT score landscape", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
ax.axvspan(500, 800, alpha=0.05, color='red')
ax.axvspan(0, 500, alpha=0.05, color='green')

# --- Bottom: inversion magnitude -------------------------------------------
ax = axes2[2]
ax.fill_between(t_years, inv_magnitudes, alpha=0.4, color='firebrick')
ax.plot(t_years, inv_magnitudes, lw=1.2, color='firebrick')
ax.axvline(best_no_pen, color='steelblue', lw=1.5, ls='--')
ax.axvline(best_pen,    color='firebrick', lw=1.5, ls='--')
ax.set_xlabel("Window start  (years)", fontsize=9)
ax.set_ylabel("Inversion\nmagnitude", fontsize=9)
ax.set_title("Topological inversion magnitude per window", fontsize=10)
ax.axvspan(500, 800, alpha=0.05, color='red')
ax.axvspan(0, 500, alpha=0.05, color='green')

fig2.suptitle(
    "Effect of Topological Penalty on Window Selection\n"
    "Without penalty, a phase-inverted window can score as well as the correct one.",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.show()

# %%
# -------------------------------------------------------------------------
# SECTION 6: Score sensitivity to seq_weight
# -------------------------------------------------------------------------
# How strongly does the penalty need to be weighted to reliably select the
# correct (phase-matched) window?

seq_weights  = np.linspace(0, 5, 60)
best_window_yr = []

for sw in seq_weights:
    s = scores_no_pen + sw * inv_magnitudes
    best_window_yr.append(t_years[np.nanargmin(s)])

best_window_yr = np.array(best_window_yr)

# Find critical seq_weight where best window shifts to phase-matched region
phase_matched_mask   = best_window_yr < 400   # heuristic: phase-matched starts ~t=0
critical_sw_idx = np.argmax(phase_matched_mask)   # first True

fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.plot(seq_weights, best_window_yr, 'o-', ms=4, lw=1.5, color='darkslategray')
ax3.axhspan(0, 400,  alpha=0.12, color='green', label='Phase-matched region (correct)')
ax3.axhspan(400, 800, alpha=0.12, color='red',  label='Phase-inverted region (wrong)')

if critical_sw_idx > 0:
    crit_sw = seq_weights[critical_sw_idx]
    ax3.axvline(crit_sw, color='black', ls='--', lw=1.5,
                label=f'Critical seq_weight ≈ {crit_sw:.2f}')
    ax3.annotate(f'seq_weight ≥ {crit_sw:.2f}\ncorrectly selects\nphase-matched window',
                 xy=(crit_sw, best_window_yr[critical_sw_idx]),
                 xytext=(crit_sw + 0.3, 500),
                 fontsize=9, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen'))

ax3.set_xlabel("seq_weight", fontsize=10)
ax3.set_ylabel("Best-fit window start (yr)", fontsize=10)
ax3.set_title("Best-fit window vs. seq_weight\n"
              "Too small → wrong (inverted) window wins;  Large enough → correct window wins",
              fontsize=10)
ax3.legend(fontsize=9)
ax3.set_ylim(-30, 830)
plt.tight_layout()
plt.show()

# %%
# -------------------------------------------------------------------------
# SECTION 7: Summary
# -------------------------------------------------------------------------
print("=" * 65)
print("SUMMARY: Topological Sequence Penalty")
print("=" * 65)
print("""
Mechanism
---------
1. Compute unbalanced OT transport plan  P[i,j]  (history→simulation).
2. For each historical event i, compute its "expected simulation index":

       <idx>_i = Σ_j  P[i,j] · j  /  Σ_j  P[i,j]

3. Compute backward jumps in the <idx> sequence:

       inversion_magnitude = Σ_i  max(0, <idx>_i - <idx>_{i+1})

4. Add a penalty term:

       total_score = base_OT_score + seq_weight × inversion_magnitude

Why it matters
--------------
• When the time axis is compressed (large scale_t), spatial proximity
  dominates OT cost.  A simulation segment starting at the "wrong"
  spatial phase can match historical events well in space but
  OUT OF ORDER in time — producing temporal inversions.

• The penalty raises the score of such inverted windows, causing the
  sliding-window search to prefer the physically correct (ordered) window.

Choosing seq_weight
-------------------
• seq_weight = 0.0  →  no penalty (pure OT, inversions allowed).
• Small seq_weight  →  gentle nudge; may still prefer inverted windows.
• Large seq_weight  →  strong prior for temporal ordering; risk of
                        over-constraining noisy / sparse data.
• Recommended: sweep seq_weight on synthetic sanity-check catalogs
  (as in Section 6) and pick the critical value where the correct
  window is reliably selected.
""")

# %%
# =============================================================================
# SECTION 8: Rupture-trace transport-plan plot — 3 exemplary cases
# =============================================================================
# Style matches the notebook visualization: horizontal bars = rupture extents
# in (longitude, time) space; coloured lines = OT connections.
# Three side-by-side panels show:
#   Case 1 — Perfect match       : low base, low penalty
#   Case 2 — Phase-inverted      : moderate base, HIGH penalty (inversions)
#   Case 3 — Bad (wrong pattern) : high base, lower penalty
# %%
import matplotlib.cm as cm
from eqcycles.analysis.synthetic import generate_linear_fault, create_event_catalog
from eqcycles.analysis.scoring   import prepare_event_data, normalize_coords, get_transport_plan

# ---- Fault geometry ---------------------------------------------------------
_SHP = '/tmp/topo_penalty_demo_fault.shp'
fault = generate_linear_fault(29, 40, 39, 40)   # ~830 km, E–W
fault.to_file(_SHP)

# ---- OT configuration (match notebook defaults) ----------------------------
CFG = dict(reg=0.1, reg_m=1.0, scale_x=100_000.0, scale_t=100.0, scale_mass=100_000.0)
SW  = 2.0   # seq_weight for the penalty term

# ---- Historical record: East → West → East pattern -------------------------
# (center_km is measured from the western tip of the fault)
HIST_EVENTS = [
    (  0,  680, 250),   # H0 : eastern segment, t=0 yr
    ( 60,  110, 220),   # H1 : western segment, t=60 yr
    (120,  675, 245),   # H2 : eastern segment, t=120 yr
]
hist_df = create_event_catalog(fault, HIST_EVENTS)

# ---- Three simulation windows -----------------------------------------------
# Case 1 – Phase-MATCHED (East→West→East, noisy copy of history)
SIM1_EVENTS = [
    (  2,  685, 255),   # ≈ H0 (east)
    ( 63,  105, 215),   # ≈ H1 (west)
    (118,  670, 240),   # ≈ H2 (east)
]

# Case 2 – Phase-INVERTED (West→East→West — opposite spatial phase)
SIM2_EVENTS = [
    (  0,  110, 250),   # S0 : west  ← H0 is east  → INVERSION
    ( 60,  680, 250),   # S1 : east  ← H1 is west  → INVERSION
    (120,  110, 245),   # S2 : west  ← H2 is east  → INVERSION
]

# Case 3 – Bad match (full-fault ruptures, different pattern entirely)
SIM3_EVENTS = [
    ( 10,  415, 680),   # S0 : full fault
    ( 75,  280, 350),   # S1 : western third
    (105,  560, 350),   # S2 : eastern third
    (140,  415, 450),   # S3 : broad central (extra event)
]

case_sims   = [SIM1_EVENTS, SIM2_EVENTS, SIM3_EVENTS]
case_labels = [
    "Case 1 — Phase-matched\n(correct window)",
    "Case 2 — Phase-inverted\n(spatial mismatch + inversions)",
    "Case 3 — Bad match\n(wrong spatial pattern)",
]

# ---- Helper: compute scores -------------------------------------------------
def _ot_scores(hist_df, sim_df, cfg, shapefile, sw):
    """Return (base_score, penalty, P, hist_norm, sim_norm, sim_df_out)."""
    hc, hm = prepare_event_data(hist_df, shapefile)
    sc, sm = prepare_event_data(sim_df,  shapefile)

    hc_n = normalize_coords(hc.copy(), cfg['scale_x'], cfg['scale_t'])
    hc_n[:, 1] -= hc_n[:, 1].min()
    hm_n = hm / cfg['scale_mass']

    sc_n = normalize_coords(sc.copy(), cfg['scale_x'], cfg['scale_t'])
    sc_n[:, 1] -= sc_n[:, 1].min()
    sm_n = sm / cfg['scale_mass']

    P = get_transport_plan(hc_n, hm_n, sc_n, sm_n, cfg)

    M = ot.dist(hc_n, sc_n, metric='euclidean')
    base = float((P * M).sum())

    # inversion magnitude
    sim_idx    = np.arange(P.shape[1], dtype=float)
    row_sums   = P.sum(axis=1)
    safe_rows  = np.where(row_sums > 1e-12, row_sums, 1.0)
    exp_idx    = P @ sim_idx / safe_rows
    steps      = np.diff(exp_idx)
    inv_mag    = float(np.maximum(0, -steps).sum())

    return base, inv_mag * sw, P, hc_n, sc_n, exp_idx, sim_df

# ---- Build the 3 cases -------------------------------------------------------
cases = []
for ev in case_sims:
    sim_df = create_event_catalog(fault, ev)
    result = _ot_scores(hist_df, sim_df, CFG, _SHP, SW)
    cases.append(result)

# ---- Print score table -------------------------------------------------------
print(f"\n{'Case':<35}  {'Base':>8}  {'Penalty':>8}  {'Total':>8}")
print("-" * 65)
for lbl, (base, pen, *_) in zip(case_labels, cases):
    short = lbl.split('\n')[0]
    print(f"{short:<35}  {base:8.4f}  {pen:8.4f}  {base+pen:8.4f}")

# ---- Main figure: 3 panels --------------------------------------------------
fig_s8, axes_s8 = plt.subplots(1, 3, figsize=(18, 8), sharey=False)

for ax, lbl, (base, pen, P, hc_n, sc_n, exp_idx, sim_df) in zip(
        axes_s8, case_labels, cases):

    sim_centers_lon = (sim_df.lon_start + sim_df.lon_end) / 2
    sim_times_yr    = sim_df.time.values

    hist_centers_lon = (hist_df.lon_start + hist_df.lon_end) / 2
    # Shift historical so that it starts at t=0 (same origin as simulation window)
    hist_t0          = hist_df.time.min()
    hist_times_yr    = hist_df.time.values - hist_t0

    # --- Simulation ruptures (gray thick bars) ---
    for _, row in sim_df.iterrows():
        ax.plot([row.lon_start, row.lon_end], [row.time, row.time],
                color='gray', alpha=0.55, linewidth=5,
                solid_capstyle='butt', zorder=2)
    # Label just the first
    ax.plot([], [], color='gray', linewidth=5, label='Simulation', alpha=0.55)

    # --- Historical ruptures (black thin bars) ---
    for _, row in hist_df.iterrows():
        t_vis = row.time - hist_t0
        ax.plot([row.lon_start, row.lon_end], [t_vis, t_vis],
                color='black', linewidth=2.5,
                solid_capstyle='butt', zorder=3)
    ax.plot([], [], color='black', linewidth=2.5, label='Historical')

    # --- OT connections (LineCollection, cool cmap) ---
    threshold = 1e-3 * P.max()
    segs, weights_c = [], []
    for i in range(len(hist_df)):
        for j in range(len(sim_df)):
            w = P[i, j]
            if w > threshold:
                segs.append([(hist_centers_lon.iloc[i], hist_times_yr[i]),
                              (sim_centers_lon.iloc[j],  sim_times_yr[j])])
                weights_c.append(w)

    if segs:
        weights_c = np.array(weights_c)
        norm_w    = mcolors.Normalize(vmin=weights_c.min(), vmax=weights_c.max())
        rgba      = cm.cool(norm_w(weights_c))
        alpha_min, alpha_max = 0.15, 1.0
        lw_min,    lw_max    = 0.5,  2.5
        if weights_c.max() > weights_c.min():
            nw = (weights_c - weights_c.min()) / (weights_c.max() - weights_c.min())
            rgba[:, 3]  = alpha_min + nw * (alpha_max - alpha_min)
            lws = lw_min + nw * (lw_max - lw_min)
        else:
            rgba[:, 3] = alpha_max
            lws = np.full(len(weights_c), lw_max)

        lc = LineCollection(segs, colors=rgba, linewidths=lws, zorder=4)
        ax.add_collection(lc)
        sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=norm_w)
        sm.set_array([])
        cbar = fig_s8.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Mass transported', fontsize=8)

    # --- Expected-index markers & inversion arrows ---------------------------
    for i, (hclon, ht, ei) in enumerate(zip(hist_centers_lon, hist_times_yr, exp_idx)):
        ax.annotate(f'H{i}\n⟨s⟩={ei:.1f}',
                    xy=(hclon, ht),
                    xytext=(hclon + 0.12, ht),
                    fontsize=7.5, va='center', color='black',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    steps = np.diff(exp_idx)
    for i, step in enumerate(steps):
        if step < 0:
            mid_t = (hist_times_yr[i] + hist_times_yr[i + 1]) / 2
            ax.annotate(
                f'⚠ Δ⟨s⟩={step:.1f}',
                xy=(hist_centers_lon.iloc[i], mid_t),
                xytext=(hist_centers_lon.iloc[i] - 1.2, mid_t),
                fontsize=8, color='firebrick', va='center',
                arrowprops=dict(arrowstyle='->', color='firebrick', lw=0.9)
            )

    # --- Axes formatting ---
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Time (years)", fontsize=10)
    total = base + pen
    title = (f"{lbl}\n"
             f"Base OT = {base:.4f}   "
             f"Penalty = {pen:.4f}   "
             f"Total = {total:.4f}")
    ax.set_title(title, fontsize=9, pad=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(28.5, 40.5)

fig_s8.suptitle(
    f"Transport-Plan Comparison — OT Base Score vs. Penalized Score\n"
    f"(reg={CFG['reg']}, reg_m={CFG['reg_m']}, seq_weight={SW}  |  "
    f"scale_t={CFG['scale_t']} yr → space dominates OT cost)",
    fontsize=11, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.show()
