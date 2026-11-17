"""Figure 1: what the first token says, and what the model then writes.

Three interventions on the same false-positive films (every intervention is in Table 1). For each,
two bars. Top: the probability of the first output token, split into "yes", "no" and every other
token (almost all "Based", the start of a reasoned answer); P_norm = yes / (yes + no) is printed at
right. Bottom: the answers the model writes when sampled at temperature 1, split by the first word:
answers that open with "Yes" or "No" (dark), answers that reason first (light), and answers that
never say yes or no (white); the sampled yes rate s is printed at right. The figure is
drawn at its printed width so its text is not scaled down.

Reads E4 (``readout_sampled_answers.json``, groups fp_banked + fp_fresh) when present, else E2
(``readout_generation_check.json``, the 20 banked films).

Times to match the body text. Yes in greys, no in red (ColorBrewer Reds), so the eye goes to the
"no" answers.
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

# dark: the answer itself, light: the same answer after reasoning
YES, YES2, NO, NO2, OTHER = "#4d4d4d", "#b3b3b3", "#cb181d", "#fc9272", "#dedede"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42})
R = Path("results/clt_scale")

# Row names match Table 1.
ARMS = [("No\nintervention", "base"),
        ("Suppress,\nscale $-1$", "clamp_scale-1"),
        ("Evidence block,\nblack", "black")]
ALL_ARMS = ["base", "clamp_scale-1", "clamp_ablate+0", "clamp_scale-2", "blur", "black", "border"]


def load() -> tuple[list[dict], str]:
    """Per-film ``{arm: record}`` with ``samples_open`` filled in, and a source label."""
    e4 = R / "readout_sampled_answers.json"
    if e4.exists():
        rows = [r["arms"] for r in json.load(open(e4))["rows"]
                if r["group"] in ("fp_banked", "fp_fresh")]
        return rows, "E4"
    rows = []
    for r in json.load(open(R / "readout_generation_check.json"))["rows"]:
        arms = {k: dict(r[k]) for k in ALL_ARMS}
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
    fig, ax = plt.subplots(figsize=(4.45, 2.25))  # drawn at its printed size
    H, PAIR, STEP = 0.36, 0.42, 1.25
    ticks, ticklabels = [], []
    for i, (name, key) in enumerate(ARMS):
        recs = [r[key] for r in rows]
        yt = -i * STEP
        yb = yt - PAIR
        yes, no = first_token(recs)
        ax.barh(yt, yes, color=YES, height=H, lw=0)
        ax.barh(yt, no, left=yes, color=NO, height=H, lw=0)
        ax.barh(yt, 1 - yes - no, left=yes + no, color=OTHER, height=H, lw=0)
        w = written(recs)
        left = 0.0
        for k, fc in (("yes_direct", YES), ("yes_reasoned", YES2), ("no_reasoned", NO2),
                      ("no_direct", NO), ("none", "white")):
            ax.barh(yb, w[k], left=left, color=fc, height=H, lw=0)
            left += w[k]
        # one thin frame round the written bar, so its white "neither" end is visible
        ax.barh(yb, 1, color="none", edgecolor="0.55", height=H, lw=0.4)
        s = w["yes_direct"] + w["yes_reasoned"]
        norm = float(np.mean([a["norm"] for a in recs]))
        ax.text(1.02, yt, f"$P_{{\\mathrm{{norm}}}}={norm:.3f}$", va="center")
        ax.text(1.02, yb, f"$s={s:.2f}$", va="center")
        ax.annotate(name, xy=(0, (yt + yb) / 2), xycoords=("axes fraction", "data"),
                    xytext=(-45, 0), textcoords="offset points", ha="right", va="center",
                    linespacing=1.1)
        ticks += [yt, yb]
        ticklabels += ["first token", "written"]
    ax.set_yticks(ticks, ticklabels)
    ax.set_ylim(-(len(ARMS) - 1) * STEP - PAIR - 0.35, 0.35)
    ax.set_xlim(0, 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(f"fraction (mean over {n_films} films)")
    key = [Patch(fc=YES, label="yes"), Patch(fc=YES2, label="yes, after reasoning"),
           Patch(fc=NO, label="no"), Patch(fc=NO2, label="no, after reasoning"),
           Patch(fc=OTHER, label="other first token"),
           Patch(fc="white", ec="0.55", lw=0.4, label="neither")]
    ax.legend(handles=key, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0)
    fig.savefig("paper/figures/fig1_split.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("paper/figures/fig1_split.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote fig1_split from {src} ({n_films} films)")


if __name__ == "__main__":
    main()
