"""Figure: film by film, what the normalized readout says against what the model writes.

One point per film and intervention on the 40 false-positive films of E4
(``readout_sampled_answers.json``, groups fp_banked + fp_fresh). x: the sampled yes rate s after the
edit. y: P_norm after the edit. Colour and marker: the greedy answer (dark grey circle yes, red square
no, light grey triangle neither), as in Figure 1. (a) The three suppression variants. (b) The three
occlusions of the evidence block (black, mean fill, blur). A red point above the dashed line is
a greedy "no" on which P_norm still says yes; the caption counts them with pf.clamp.gnoyes /
pf.clamp.gno and pf.occ.gnoyes / pf.occ.gno of make_numbers.py, which the printout below checks.

Drawn at its printed width so its text is not scaled down.
Run from the repo root: .venv/bin/python paper/figures/make_fig_films.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

YES, NO, NONE = "#4d4d4d", "#cb181d", "#b3b3b3"  # as in Figure 1: yes grey, no red
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42})
R = Path("results/clt_scale")

PANELS = [("(a) Suppress the top 80 features", ("clamp_scale-1", "clamp_ablate+0", "clamp_scale-2")),
          ("(b) Occlude the evidence block", ("black", "mean", "blur"))]
STYLE = {"YES": (YES, "o", "yes"), "NO": (NO, "s", "no"), "NONE": (NONE, "^", "neither")}


def s_rate(a: dict) -> float:
    p = a["samples_parsed"]
    return sum(x == "YES" for x in p) / len(p)


def main() -> None:
    rows = [r["arms"] for r in json.load(open(R / "readout_sampled_answers.json"))["rows"]
            if r["group"] in ("fp_banked", "fp_fresh")]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=True)  # drawn at its printed size
    for ax, (title, arms) in zip(axes, PANELS, strict=True):
        recs = [r[a] for r in rows for a in arms]
        for ans in ("YES", "NONE", "NO"):  # "no" last, so it is drawn on top
            pts = [(s_rate(a), a["norm"]) for a in recs if a["greedy_parsed"] == ans]
            if pts:
                col, mk, name = STYLE[ans]
                xs, ys = zip(*pts, strict=True)
                ax.scatter(xs, ys, s=10, marker=mk, color=col, lw=0, alpha=0.75, label=name)
        ax.axhline(0.5, color="0.4", lw=0.6, ls="--", zorder=0)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.04, 1.04)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("sampled yes rate $s$")
    axes[0].set_ylabel("$P_{\\mathrm{norm}}$")
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(k) for k in ("yes", "no", "neither")]
    leg = axes[0].legend([h[i] for i in order], [lab[i] for i in order], title="greedy answer",
                         loc="lower right", fontsize=7, title_fontsize=7, handletextpad=0.2,
                         fancybox=False, edgecolor="0.7", framealpha=1)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig("paper/figures/fig_films.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("paper/figures/fig_films.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    for (title, arms) in PANELS:
        recs = [r[a] for r in rows for a in arms]
        n_no = sum(a["greedy_parsed"] == "NO" for a in recs)
        n_miss = sum(a["greedy_parsed"] == "NO" and a["norm"] >= 0.5 for a in recs)
        print(f"{title}: P_norm >= 0.5 on {n_miss} of {n_no} greedy no answers")
    print(f"wrote fig_films ({len(rows)} films)")


if __name__ == "__main__":
    main()
