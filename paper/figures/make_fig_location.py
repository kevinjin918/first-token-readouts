"""Figure: where the occlusion evidence sits on the film.

(a) For each of the 20 AP films the model calls cardiomegaly-positive (all labeled negative), the
    block whose black occlusion most lowers P(yes) (``block_effect`` in attn_causal_map, a P_raw
    drop), counted on the 8x8 block grid over the mean of the 20 films. (b) One of those films
    with its block outlined. (c) The most damaging block for the effusion- and
    pneumothorax-positive films where one block is removable (P_norm drop > 0.15,
    ``block_effect`` in causal_map_general).

Times to match the body text; red as in Figures 1 and 2.
Needs the NIH images under ~/.cache/tracecxr/data/chestxray14/images.
Run from the repo root: .venv/bin/python paper/figures/make_fig_location.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RED = "#cb181d"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "pdf.fonttype": 42})
NB, SIDE = 8, 512
IMG = Path.home() / ".cache/tracecxr/data/chestxray14/images"
R = "results/clt_scale/"

rows = json.load(open(R + "occlusion_readout_audit.json"))["rows"]
films = [np.asarray(Image.open(IMG / r["image"]).convert("L").resize((SIDE, SIDE)), float)
         for r in rows]
mean_film = np.mean(films, axis=0)
counts = np.zeros((NB, NB), int)
for r in rows:
    counts[divmod(r["top_block"], NB)] += 1


def removable(name):
    out = []
    for rec in json.load(open(R + f"causal_map_{name}.json"))["records"]:
        be = np.asarray(rec["block_effect"])
        if be.max() > 0.15:
            out.append(divmod(int(be.argmax()), NB))
    return out


eff, ptx = removable("effusion"), removable("pneumothorax")
cell = SIDE / NB

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.1})
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)

ax = axes[0]
ax.imshow(mean_film, cmap="gray", vmin=0, vmax=255)
for (i, j), c in np.ndenumerate(counts):
    if c:
        ax.add_patch(Rectangle((j * cell, i * cell), cell, cell, fc=RED, ec="white", lw=0.4,
                               alpha=min(0.35 + 0.12 * c, 0.9)))
        ax.text((j + 0.5) * cell, (i + 0.5) * cell, str(c), color="white", ha="center",
                va="center", fontsize=7)
ax.set_title("(a) Cardiomegaly calls", fontsize=7.5)
ax.set_xlabel("films per block ($n=20$)")

ax = axes[1]
ex = rows[0]
ax.imshow(films[0], cmap="gray", vmin=0, vmax=255)
i, j = divmod(ex["top_block"], NB)
ax.add_patch(Rectangle((j * cell, i * cell), cell, cell, fc="none", ec=RED, lw=1.2))
ax.set_title("(b) One of the 20", fontsize=7.5)
ax.set_xlabel(f"outlined block blacked out:\n$P_{{\\mathrm{{norm}}}}$ {ex['base']['norm']:.2f}"
              f" $\\to$ {ex['black']['norm']:.2f}", linespacing=1.1)

ax = axes[2]
ax.imshow(mean_film, cmap="gray", vmin=0, vmax=255, alpha=0.85)
for pts, col, mk, lab in ((eff, RED, "o", f"effusion (n={len(eff)})"),
                          (ptx, "white", "^", f"pneumothorax (n={len(ptx)})")):
    jit = np.random.default_rng(len(pts)).uniform(-0.18, 0.18, (len(pts), 2))
    ax.scatter([(c + 0.5 + dx) * cell for (_, c), (dx, _) in zip(pts, jit, strict=True)],
               [(r + 0.5 + dy) * cell for (r, _), (_, dy) in zip(pts, jit, strict=True)],
               s=20, marker=mk, color=col, edgecolor="black", lw=0.4, label=lab, zorder=3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, frameon=False,
          handletextpad=0.1, labelspacing=0.2)
ax.set_title("(c) Effusion, pneumothorax", fontsize=7.5)

fig.savefig("paper/figures/fig_location.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("paper/figures/fig_location.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_location; counts by row", counts.sum(1).tolist(), "by col", counts.sum(0).tolist())
