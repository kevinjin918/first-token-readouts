"""Figure: the features each film's edit suppresses, across four groups of films.

One row per film of E4 (``readout_sampled_answers.json``): the 40 cardiomegaly false positives and
the 20 correct calls each for cardiomegaly, effusion and pneumothorax. A dark cell is one of the 80
features suppressed on that film (``clamp_pairs``, the film's top 80 by mean activation over the image
tokens). Columns are the transcoder features any film's edit uses, ordered by how many films use
them, followed (right of the dashed line) by the features only the contrast sets use. The lower
panel holds the contrast sets of E5 (``clamp_contrastive_answers.json``, ``sets``): the 80 features
whose mean activation differs most between positive and negative donors, for cardiomegaly and for
effusion. The caption's counts are feat.* of make_numbers.py; the printout below checks them.

Drawn at its printed width so its text is not scaled down.
Run from the repo root: .venv/bin/python paper/figures/make_fig_features.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42})
R = Path("results/clt_scale")

GROUPS = [("false positives, cardiomegaly", ("fp_banked", "fp_fresh")),
          ("correct, cardiomegaly", ("tp_cardiomegaly",)),
          ("correct, effusion", ("tp_effusion",)),
          ("correct, pneumothorax", ("tp_pneumothorax",))]
CONTRAST = [("contrast set, cardiomegaly", "con_card"), ("contrast set, effusion", "con_eff")]
BINARY = ListedColormap(["white", "black"])


def main() -> None:
    rows = json.load(open(R / "readout_sampled_answers.json"))["rows"]
    films = [[{tuple(f) for f in r["clamp_pairs"]} for r in rows if r["group"] in gs]
             for _, gs in GROUPS]
    sets = json.load(open(R / "clamp_contrastive_answers.json"))["sets"]
    con = [{tuple(f) for f in sets[k]} for _, k in CONTRAST]
    n_films = sum(len(g) for g in films)
    count = Counter(f for g in films for s in g for f in s)
    pool = sorted(count, key=lambda f: (-count[f], f))
    only = lambda a, b: sorted(a - b - set(count))  # noqa: E731
    extra = only(con[0], con[1]) + sorted((con[0] & con[1]) - set(count)) + only(con[1], con[0])
    cols = pool + extra
    every = sum(c == n_films for c in count.values())
    member = lambda sets_: np.array([[f in s for f in cols] for s in sets_], int)  # noqa: E731

    fig, (top, bot) = plt.subplots(2, 1, figsize=(5.0, 2.45), sharex=True,  # its printed size
                                   gridspec_kw={"height_ratios": [n_films, 16], "hspace": 0.08})
    rows_ = [s for g in films for s in g]
    top.imshow(member(rows_), cmap=BINARY, aspect="auto", interpolation="none",
               extent=(0, len(cols), n_films, 0))
    edges = np.cumsum([0] + [len(g) for g in films])
    for e in edges[1:-1]:
        top.axhline(e, color="0.5", lw=0.6)
    top.set_yticks((edges[:-1] + edges[1:]) / 2,
                   [f"{name} ({len(g)})" for (name, _), g in zip(GROUPS, films, strict=True)])
    top.tick_params(axis="y", length=0)
    bot.imshow(member(con), cmap=BINARY, aspect="auto", interpolation="none",
               extent=(0, len(cols), 2, 0))
    bot.axhline(1, color="0.5", lw=0.6)
    bot.set_yticks([0.5, 1.5], [name for name, _ in CONTRAST])
    bot.tick_params(axis="y", length=0)
    for ax in (top, bot):
        ax.axvline(len(pool), color="0.3", lw=0.7, ls="--")
    bot.set_xlabel("feature, ordered by the number of films whose edit uses it")
    fig.savefig("paper/figures/fig_features.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("paper/figures/fig_features.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote fig_features ({n_films} films, {len(pool)} pool features, {every} on every "
          f"film, {len(extra)} contrast-only)")


if __name__ == "__main__":
    main()
