"""Figure 2: generalisation across findings and detection regimes.

(a) Spearman rho between a block's attention weight and its own causal effect, per film, against a
    shuffled-attention null. (b) mean effect of the single most damaging block, with the count of
    films where that block moves normalised P by >0.15.

Palette: dataviz categorical slot 1 (#2a78d6); validated colourblind-safe against the light surface.
Run from the repo root: .venv/bin/python paper/figures/make_fig2.py
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SER, INK, INK2, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#8f8e89", "#d5d4cf"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "axes.edgecolor": GRID, "axes.linewidth": 0.6})
REMX, XMAX = 0.81, 0.96

CELLS = [("Cardiomegaly", "false positive, AP-only", "results/clt_scale/causal_map_card_neg_AP.json"),
         ("Cardiomegaly", "false positive, any view", "results/clt_scale/causal_map_cardiomegaly_neg.json"),
         ("Cardiomegaly", "genuine detection", "results/clt_scale/causal_map_cardiomegaly_pos.json"),
         ("Pneumothorax", "genuine detection", "results/clt_scale/causal_map_pneumothorax.json"),
         ("Effusion",     "genuine detection", "results/clt_scale/causal_map_effusion.json")]
data = []
for fin, sel, fn in CELLS:
    recs = json.load(open(fn))["records"]
    rho = np.array([r["rho"] for r in recs]); sh = np.array([r["rho_shuffled"] for r in recs])
    best = np.array([r["block_effect"] for r in recs]).max(axis=1)
    data.append(dict(fin=fin, sel=sel, rho=rho[~np.isnan(rho)], shuf=np.nanmean(sh),
                     best=float(best.mean()), rem=int((best > 0.15).sum()), n=len(recs)))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.9, 2.9),
                               gridspec_kw={"width_ratios": [1.18, 1], "wspace": 0.30})
N = len(data)
axA.axvspan(0.3, 1.0, color=GRID, alpha=0.45, zorder=0, lw=0)
axA.text(0.985, 0.965, "pre-specified\n“attention explains”", transform=axA.transAxes,
         fontsize=6.5, color=MUTED, va="top", ha="right", linespacing=1.25)
axA.axvline(0, color=MUTED, lw=0.8, zorder=1)
rng = np.random.default_rng(0)
tr = axA.get_yaxis_transform()
for i, d in enumerate(data):
    y = N - 1 - i
    axA.scatter(d["rho"], y + rng.uniform(-0.10, 0.10, len(d["rho"])), s=9, color=SER,
                alpha=0.30, lw=0, zorder=2)
    axA.scatter([d["shuf"]], [y], s=42, facecolor="#fcfcfb", edgecolor=MUTED, lw=1.2, zorder=3)
    axA.scatter([d["rho"].mean()], [y], s=48, color=SER, edgecolor="#fcfcfb", lw=1.2, zorder=4)
    axA.text(-0.025, y + 0.16, d["fin"], transform=tr, ha="right", va="center",
             fontsize=8, color=INK, clip_on=False)
    axA.text(-0.025, y - 0.17, d["sel"], transform=tr, ha="right", va="center",
             fontsize=6.8, color=MUTED, style="italic", clip_on=False)
axA.set_xlim(-0.55, 0.72); axA.set_ylim(-0.60, N - 0.35); axA.set_yticks([])
axA.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
axA.xaxis.grid(True, color=GRID, lw=0.4, zorder=0); axA.set_axisbelow(True)
for s in ("top", "right", "left"): axA.spines[s].set_visible(False)
axA.tick_params(axis="x", colors=INK2, length=0, pad=2)
axA.set_xlabel(r"Spearman $\rho$ (block attention vs its causal effect)", fontsize=7.8,
               color=INK2, labelpad=3)
axA.set_title("a   Attention weight vs. causal effect", fontsize=8.5, color=INK,
              loc="left", pad=20, fontweight="bold")
for h, l in ((dict(s=9, color=SER, alpha=0.4, lw=0), "per film"),
             (dict(s=48, color=SER), "mean"),
             (dict(s=42, facecolor="#fcfcfb", edgecolor=MUTED, lw=1.2), "shuffled null")):
    axA.scatter([], [], label=l, **h)
axA.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=3, frameon=False, fontsize=7,
           labelcolor=INK2, borderpad=0, handletextpad=0.35, columnspacing=1.1)

for i, d in enumerate(data):
    y = N - 1 - i
    w, r = d["best"], 0.010
    axB.add_patch(FancyBboxPatch((0, y - 0.15), max(w - r, 1e-4), 0.30,
                                 boxstyle=f"round,pad=0,rounding_size={r}",
                                 fc=SER, ec="none", mutation_aspect=0.18, zorder=3))
    axB.text(w + 0.020, y, f"{w:.3f}", va="center", ha="left", fontsize=7.4, color=INK)
    axB.text(REMX, y, f"{d['rem']}/{d['n']}", va="center", ha="center", fontsize=7.4, color=INK2)
axB.text(REMX, 1.02, "films with a\nremovable block", transform=axB.get_xaxis_transform(),
         va="bottom", ha="center", fontsize=6.5, color=MUTED, linespacing=1.25)
axB.set_xlim(0, XMAX); axB.set_ylim(-0.60, N - 0.35); axB.set_yticks([])
axB.set_xticks([0, 0.2, 0.4])
axB.xaxis.grid(True, color=GRID, lw=0.4, zorder=0); axB.set_axisbelow(True)
for s in ("top", "right", "left"): axB.spines[s].set_visible(False)
axB.tick_params(axis="x", colors=INK2, length=0, pad=2)
axB.set_xlabel("mean effect of the single best block", fontsize=7.8, color=INK2, labelpad=3)
axB.set_title("b   Removability is conditional", fontsize=8.5, color=INK,
              loc="left", pad=20, fontweight="bold")

fig.savefig("paper/figures/fig2_generalisation.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("paper/figures/fig2_generalisation.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote fig2")
