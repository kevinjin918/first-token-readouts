"""Figure 1: what the first token says, and what the model then writes.

For each intervention, two stacked bars over the same false-positive films. Top: the probability
of the first output token, split into "yes", "no" and every other token (almost all "Based", the
start of a reasoned answer). P_raw is the yes segment and P_norm is yes / (yes + no). Bottom: the
answers the model writes when sampled at temperature 1, split by the first word: answers that
open with "Yes" or "No" (solid) and answers that reason first (hatched), and answers that never
say yes or no (grey). Right: the films whose greedy answer is no, and among them the films on
which P_norm still calls yes (P_norm >= 0.5).

Reads E4 (``readout_sampled_answers.json``, groups fp_banked + fp_fresh) when present, else E2
(``readout_generation_check.json``, the 20 banked films).

Palette: dataviz categorical slots 1-2 (#2a78d6, #eb6834) plus a neutral; colourblind-safe.
Run from the repo root: .venv/bin/python paper/figures/make_fig_split.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracecxr.intervention.generation_check import opening_word  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

YES, NO, OTHER, NONE = "#2a78d6", "#eb6834", "#c9c8c2", "#f4f3ef"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8f8e89"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "hatch.linewidth": 0.6})
R = Path("results/clt_scale")

# Row names match Table 1.
ARMS = [("No intervention", "base"),
        ("Suppress, scale $-1$", "clamp_scale-1"),
        ("Suppress, ablate", "clamp_ablate+0"),
        ("Suppress, scale $-2$", "clamp_scale-2"),
        ("Evidence block, blur", "blur"),
        ("Evidence block, black", "black"),
        ("Border block, black", "border")]


def load() -> tuple[list[dict], str]:
    """Per-film ``{arm: record}`` with ``samples_open`` filled in, and a source label."""
    e4 = R / "readout_sampled_answers.json"
    if e4.exists():
        rows = [r["arms"] for r in json.load(open(e4))["rows"]
                if r["group"] in ("fp_banked", "fp_fresh")]
        return rows, "E4"
    rows = []
    for r in json.load(open(R / "readout_generation_check.json"))["rows"]:
        arms = {k: dict(r[k]) for _, k in ARMS}
        for a in arms.values():
            a["samples_open"] = [opening_word(t) for t in a["samples"]]
        rows.append(arms)
    return rows, "E2"


def first_token(recs: list[dict]) -> tuple[float, float]:
    raw = np.array([a["raw"] for a in recs])
    mass = np.array([a["mass"] for a in recs])
    return float(raw.mean()), float((mass - raw).mean())


def written(recs: list[dict]) -> dict[str, float]:
    """Mean over films of the fraction of samples in each (answer, route) cell."""
    cells = {k: [] for k in ("yes_direct", "yes_reasoned", "no_direct", "no_reasoned", "none")}
    for a in recs:
        n = len(a["samples_parsed"])
        c = dict.fromkeys(cells, 0)
        for p, w in zip(a["samples_parsed"], a["samples_open"], strict=True):
            if p == "NONE":
                c["none"] += 1
            else:
                ans = p.lower()
                c[f"{ans}_direct" if w in ("yes", "no") else f"{ans}_reasoned"] += 1
        for k in cells:
            cells[k].append(c[k] / n)
    return {k: float(np.mean(v)) for k, v in cells.items()}


def main() -> None:
    rows, src = load()
    n_films = len(rows)
    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    H, GAP = 0.34, 0.06
    n = len(ARMS)
    for i, (label, key) in enumerate(ARMS):
        recs = [r[key] for r in rows]
        y0 = (n - 1 - i) * 1.0
        yt, yb = y0 + (H + GAP) / 2, y0 - (H + GAP) / 2
        # first token
        yes, no = first_token(recs)
        ax.barh(yt, yes, color=YES, height=H, lw=0)
        ax.barh(yt, no, left=yes, color=NO, height=H, lw=0)
        ax.barh(yt, 1 - yes - no, left=yes + no, color=OTHER, height=H, lw=0)
        # written answers
        w = written(recs)
        left = 0.0
        for k, col, hatch in (("yes_direct", YES, None), ("yes_reasoned", YES, "////"),
                              ("no_reasoned", NO, "////"), ("no_direct", NO, None),
                              ("none", NONE, "....")):
            fc = NONE if k == "none" else "white" if hatch else col
            ec = MUTED if k == "none" else col
            ax.barh(yb, w[k], left=left, color=fc, edgecolor=ec, hatch=hatch, height=H,
                    lw=0.0 if not hatch else 0.4)
            left += w[k]
        ax.text(-0.088, y0, label, ha="right", va="center", fontsize=7.8, color=INK)
        ax.text(-0.004, yt, "1st token", ha="right", va="center", fontsize=6.2, color=MUTED)
        ax.text(-0.004, yb, "written", ha="right", va="center", fontsize=6.2, color=MUTED)
        if yes > 0.1:
            ax.text(yes / 2, yt, f"{yes:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white")
        s = w["yes_direct"] + w["yes_reasoned"]
        ax.text(s / 2, yb, f"{s:.2f}", ha="center", va="center", fontsize=6.6, color=INK)
        gno = [a for a in recs if a["greedy_parsed"] == "NO"]
        missed = sum(a["norm"] >= 0.5 for a in gno)
        norm = float(np.mean([a["norm"] for a in recs]))
        ax.text(1.065, yt, f"{norm:.3f}", ha="center", va="center", fontsize=7.6, color=INK)
        ax.text(1.175, y0, f"{len(gno)}", ha="center", va="center", fontsize=7.6, color=INK2)
        ax.text(1.285, y0, f"{missed}", ha="center", va="center", fontsize=7.6, color=INK,
                fontweight="bold")
    top = n - 0.45
    ax.text(1.065, top, "$P_{\\mathrm{norm}}$\n(1st token)", ha="center", va="bottom",
            fontsize=6.6, color=MUTED, linespacing=1.15)
    ax.text(1.175, top, "greedy\nanswer no", ha="center", va="bottom", fontsize=6.6,
            color=MUTED, linespacing=1.15)
    ax.text(1.285, top, "$P_{\\mathrm{norm}}$\nstill yes", ha="center", va="bottom",
            fontsize=6.6, color=MUTED, linespacing=1.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis="x", colors=INK2, length=0, pad=2, labelsize=7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(OTHER)
    ax.set_xlabel(f"first token: probability; written: share of samples (mean over {n_films} films)",
                  fontsize=7.4, color=INK2, labelpad=2)
    handles = [Patch(color=YES, label="yes"), Patch(color=NO, label="no"),
               Patch(color=OTHER, label="other first token (mostly “Based”)"),
               Patch(facecolor="white", edgecolor=INK2, hatch="////", lw=0.4,
                     label="written after reasoning"),
               Patch(facecolor=NONE, edgecolor=MUTED, hatch="....", lw=0.4,
                     label="no yes/no in the text")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5,
              frameon=False, fontsize=6.6, handlelength=1.1, handletextpad=0.4,
              columnspacing=0.9, borderaxespad=0.1)
    fig.savefig("paper/figures/fig1_split.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("paper/figures/fig1_split.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote fig1_split from {src} ({n_films} films)")


if __name__ == "__main__":
    main()
