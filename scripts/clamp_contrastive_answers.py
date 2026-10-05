"""At matched N = 80, does selecting features by contrast make suppression finding-specific?

The paper's suppression clamps each film's own 80 most active image features. Those sets are nearly
the same for every finding (patient-disjoint Jaccard 0.944 at N = 80,
`donor_sets_patient_disjoint`),
so the clamp cannot be specific to cardiomegaly. A reviewer asked for the decisive version: select
features by their contrast between positive and negative films, clamp the same number, and score
the effect by what the model SAYS (sampled answers), not by a first-token probability. Earlier runs
of contrast selection were scored by P_raw only (0.026 drop, `operator_sweep.md`).

TEST FILMS, from `readout_sampled_answers.json` (E4), with E4's top-80 sets and blocks:
  fp                fp_banked + fp_fresh (40 AP films, no cardiomegaly label, model says yes)
  tp_cardiomegaly   20 films with the label, model says yes
  tp_effusion       20 films with the label, model says yes

CONTRAST DONORS, drawn here: up to --n-donor films each of cardiomegaly-positive (label, greedy
YES), cardiomegaly-negative (no label, greedy NO), effusion-positive, effusion-negative. One film
per patient; no patient shared with any E4 group, E2's films, or another donor set; seeded order,
sets taking turns. Contrast score = mean image-token activation over positives minus over negatives.

ARMS, every clamp scale -1 at image positions on exactly 80 (layer, feature) pairs:
  base          no edit
  top           the film's own top 80 by mean activation (E4's set: the paper's clamp)
  con_card      top 80 by cardiomegaly contrast (one set, all films)
  con_eff       top 80 by effusion contrast (one set, all films)
  rand_act      80 drawn uniformly from the pairs with positive mean activation on this film
  anch_evid     top 80 by mean activation over the 4 tokens of the film's occlusion block
  anch_border   top 80 by mean activation over the 4 tokens of the film's border block
Per film x arm: readouts, greedy answer, --n-samples answers at T = 1 (E2's settings and parse).
Per film and set: how many of the 80 are active on the film, and the overlap with `top`.

PRE-SPECIFIED, fixed before running (commit = pre-registration). Per film, d(arm) = s_base - s_arm,
the drop in the fraction of samples that say yes. Intervals: film bootstrap, 10,000 resamples.
S1 specificity, the double dissociation. a = d(con_card) - d(con_eff) on tp_cardiomegaly,
   b = d(con_eff) - d(con_card) on tp_effusion.
     SELECTION RESCUES SPECIFICITY   mean(a) > 0.10 with interval above 0, AND the same for b.
     CONTRASTIVE SETS INERT          upper bound of d(con_card) - d(rand_act) on tp_cardiomegaly
                                     < 0.10, AND of d(con_eff) - d(rand_act) on tp_effusion < 0.10.
     MIXED                           otherwise.
S2 on the paper's population (fp), e = d(con_card) - d(rand_act):
     CONTRAST MOVES THE FALSE POSITIVE   mean(e) > 0.10 with interval above 0
     CONTRAST INERT ON FP                upper bound < 0.10
     UNRESOLVED                          otherwise
S3 location, on fp, l = d(anch_evid) - d(anch_border):
     LOCATION MATTERS      mean(l) > 0.10 with interval above 0
     LOCATION DOES NOT     upper bound < 0.10
     UNRESOLVED            otherwise
   The same S3 quantity is reported on each tp group.
Reported for every arm and group: d(arm) with its interval, mean P_raw / P_norm, and E4's readout
verdicts (P_raw and P_norm against the sampled drop, MISREADS / TRACKS / UNRESOLVED).

GPU. Usage (VM), after readout_sampled_answers.py:
  python scripts/clamp_contrastive_answers.py --e4 /mnt/fast/clt/readout_sampled_answers.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tracecxr.intervention.answer_runner import (
    LABELS,
    PROMPTS,
    block_patch_ids,
    draw_films,
    patient,
)
from tracecxr.intervention.generation_check import (
    bootstrap_ci,
    effect_verdict,
    yes_rate,
)

N_FEATS = 80
ARMS = ("base", "top", "con_card", "con_eff", "rand_act", "anch_evid", "anch_border")
TEST_GROUPS = {"fp_banked": "fp", "fp_fresh": "fp", "tp_cardiomegaly": "tp_cardiomegaly",
               "tp_effusion": "tp_effusion"}
FINDING = {"fp": "cardiomegaly", "tp_cardiomegaly": "cardiomegaly", "tp_effusion": "effusion"}
DONOR_SETS = {
    "card_pos": ("cardiomegaly", True), "card_neg": ("cardiomegaly", False),
    "eff_pos": ("effusion", True), "eff_neg": ("effusion", False),
}
MARGIN, BIG = 0.05, 0.10


def log(*a: object) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------
# analysis (pure)
# ---------------------------------------------------------------------------------------------
def drops(rows: list[dict], arm: str) -> np.ndarray:
    return np.array([yes_rate(r["arms"]["base"]["samples_parsed"])
                     - yes_rate(r["arms"][arm]["samples_parsed"]) for r in rows])


def three_way(x: np.ndarray, pos: str, neg: str, other: str, *, n_boot: int) -> tuple[str, list]:
    m, lo, hi = bootstrap_ci(x, n_boot=n_boot)
    v = pos if (m > BIG and lo > 0) else neg if hi < BIG else other
    return v, [m, lo, hi]


def analyze(rows: list[dict], *, n_boot: int = 10000) -> dict:
    by = {g: [r for r in rows if r["group"] == g] for g in FINDING}
    summary: dict = {}
    for g, rs in by.items():
        if not rs:
            continue
        summary[g] = {}
        for arm in ARMS[1:]:
            d = drops(rs, arm)
            ent: dict = {"n": len(rs), "s_drop": bootstrap_ci(d, n_boot=n_boot),
                         "raw": float(np.mean([r["arms"][arm]["raw"] for r in rs])),
                         "norm": float(np.mean([r["arms"][arm]["norm"] for r in rs]))}
            for rn in ("raw", "norm"):
                rd = np.array([r["arms"]["base"][rn] - r["arms"][arm][rn] for r in rs])
                ent[f"{rn}_verdict"] = effect_verdict(rd - d, margin=MARGIN, n_boot=n_boot)
            ent["n_active"] = float(np.mean([r["set_stats"][arm]["n_active"] for r in rs]))
            ent["overlap_top"] = float(np.mean([r["set_stats"][arm]["overlap_top"] for r in rs]))
            summary[g][arm] = ent
    v: dict = {}
    if by["tp_cardiomegaly"] and by["tp_effusion"]:
        tc, te = by["tp_cardiomegaly"], by["tp_effusion"]
        a = drops(tc, "con_card") - drops(tc, "con_eff")
        b = drops(te, "con_eff") - drops(te, "con_card")
        ia, ib = bootstrap_ci(a, n_boot=n_boot), bootstrap_ci(b, n_boot=n_boot)
        ic = bootstrap_ci(drops(tc, "con_card") - drops(tc, "rand_act"), n_boot=n_boot)
        ie = bootstrap_ci(drops(te, "con_eff") - drops(te, "rand_act"), n_boot=n_boot)
        v["S1"] = ("SELECTION RESCUES SPECIFICITY"
                   if ia[0] > BIG and ia[1] > 0 and ib[0] > BIG and ib[1] > 0 else
                   "CONTRASTIVE SETS INERT" if ic[2] < BIG and ie[2] < BIG else "MIXED")
        v["S1_detail"] = {"a": ia, "b": ib, "con_card_minus_rand": ic, "con_eff_minus_rand": ie}
    if by["fp"]:
        fp = by["fp"]
        v["S2"], v["S2_ci"] = three_way(drops(fp, "con_card") - drops(fp, "rand_act"),
                                        "CONTRAST MOVES THE FALSE POSITIVE",
                                        "CONTRAST INERT ON FP", "UNRESOLVED", n_boot=n_boot)
    for g, rs in by.items():
        if rs:
            key = "S3" if g == "fp" else f"S3_{g}"
            v[key], v[f"{key}_ci"] = three_way(drops(rs, "anch_evid") - drops(rs, "anch_border"),
                                               "LOCATION MATTERS", "LOCATION DOES NOT",
                                               "UNRESOLVED", n_boot=n_boot)
    return {"summary": summary, "verdicts": v}


def report(res: dict) -> None:
    for g, arms in res["summary"].items():
        log(f"\n=== {g} (n={next(iter(arms.values()))['n']}) ===")
        log(f"  {'arm':12s}{'s drop [95% CI]':>24s}{'P_raw':>7s}{'P_norm':>7s}"
            f"{'active':>7s}{'∩top':>6s}  raw / norm verdicts")
        for arm, a in arms.items():
            m, lo, hi = a["s_drop"]
            log(f"  {arm:12s}  {m:+.3f} [{lo:+.3f},{hi:+.3f}]{a['raw']:7.3f}{a['norm']:7.3f}"
                f"{a['n_active']:7.1f}{a['overlap_top']:6.1f}  {a['raw_verdict']} / "
                f"{a['norm_verdict']}")
    log("\n=== verdicts ===")
    for k, val in res["verdicts"].items():
        log(f"  {k}: {val}")


# ---------------------------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------------------------
def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/clt_scale"))
    ap.add_argument("--e4", type=Path, default=Path("/mnt/fast/clt/readout_sampled_answers.json"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--ckpt", default="/mnt/fast/clt/orig2048/clt_ckpt.pt")
    ap.add_argument("--n-donor", type=int, default=16)
    ap.add_argument("--max-scan", type=int, default=600)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--analyze-only", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/fast/clt/clamp_contrastive_answers.json"))
    args = ap.parse_args()

    if args.analyze_only:
        report(analyze(json.loads(args.analyze_only.read_text())["rows"]))
        return

    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from tracecxr.intervention.answer_runner import AnswerRunner  # noqa: PLC0415
    from tracecxr.intervention.generation_check import parse_answer  # noqa: PLC0415
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    e4 = json.loads(args.e4.read_text())
    tests = [r for r in e4["rows"] if r["group"] in TEST_GROUPS]
    runner = AnswerRunner.load(args.results, args.ckpt)
    img_dir = args.data_dir / "images"
    log(f"loaded MedGemma + CLT; {len(tests)} test films from {args.e4}")

    prior = json.loads(args.out.read_text()) if args.out.exists() else None
    if prior and prior.get("sets"):
        sets = {k: [tuple(p) for p in v] for k, v in prior["sets"].items()}
        log(f"resuming: contrast sets and {len(prior['rows'])} finished films from {args.out}")
    else:
        e2 = json.loads((args.results / "readout_generation_check.json").read_text())["rows"]
        used = {patient(r["image"]) for r in e2}
        used |= {patient(f["image"]) for g in e4["draw"].values() for f in g}
        df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
        have = {p.name for p in img_dir.glob("*.png") if not p.name.startswith("._")}
        df = df[df["Image Index"].isin(have)]
        pool = [(str(r["Image Index"]), str(r["Finding Labels"]), str(r["View Position"]))
                for _, r in df.iterrows()]
        order = np.random.default_rng(args.seed).permutation(len(pool))
        pool = [pool[i] for i in order]
        log(f"{len(pool)} local films; {len(used)} E2/E4 patients excluded")
        greedy_log: dict[str, str] = {}

        def accept(s: str, name: str) -> bool:
            finding, positive = DONOR_SETS[s]
            pil = Image.open(img_dir / name).convert("RGB")
            text = runner.plain_greedy(PROMPTS[finding], pil, args.max_new_tokens)
            greedy_log[f"{s}/{name}"] = text
            return parse_answer(text) == ("YES" if positive else "NO")

        def eligible(s: str):  # noqa: ANN202
            finding, positive = DONOR_SETS[s]
            return lambda lab, _v: (LABELS[finding] in lab) == positive

        donors = draw_films(pool, {s: eligible(s) for s in DONOR_SETS}, accept,
                            n=args.n_donor, used=used, max_scan=args.max_scan)
        mean = {}
        for s, names in donors.items():
            finding = DONOR_SETS[s][0]
            mean[s] = np.mean([runner.image_feats(PROMPTS[finding],
                                                  Image.open(img_dir / n).convert("RGB")).mean(0)
                               for n in names], axis=0)
            log(f"  {s}: {len(names)} donors")
        sets = {"con_card": runner.topn(mean["card_pos"] - mean["card_neg"], N_FEATS),
                "con_eff": runner.topn(mean["eff_pos"] - mean["eff_neg"], N_FEATS)}
        jac = len(set(sets["con_card"]) & set(sets["con_eff"])) / len(
            set(sets["con_card"]) | set(sets["con_eff"]))
        log(f"  contrast sets: Jaccard(card, eff) = {jac:.3f}")
        prior = {"donors": donors, "greedy_draw": greedy_log, "jaccard_con": jac,
                 "sets": {k: [[int(L), int(f)] for L, f in v] for k, v in sets.items()},
                 "rows": []}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prior, indent=1))

    rows = list(prior["rows"])
    done = {(r["group"], r["image"]) for r in rows}
    t0 = time.time()
    for k, t in enumerate(tests):
        g = TEST_GROUPS[t["group"]]
        if (g, t["image"]) in done:
            continue
        torch.manual_seed(args.seed + k)
        prompt = PROMPTS[FINDING[g]]
        pil = Image.open(img_dir / t["image"]).convert("RGB")
        F = runner.image_feats(prompt, pil)
        act = F.mean(0)
        rng = np.random.default_rng(args.seed + 7919 * k)
        live = np.argwhere(act > 0)
        pick = live[rng.choice(len(live), N_FEATS, replace=False)]
        top = [tuple(p) for p in t["clamp_pairs"]]
        film_sets = {
            "top": top, **sets,
            "rand_act": [(runner.layers[int(i)], int(j)) for i, j in pick],
            "anch_evid": runner.topn(F[block_patch_ids(t["top_block"])].mean(0), N_FEATS),
            "anch_border": runner.topn(F[block_patch_ids(t["border_block"])].mean(0), N_FEATS),
        }
        li = {L: i for i, L in enumerate(runner.layers)}
        rec = {"group": g, "source_group": t["group"], "image": t["image"],
               "top_block": t["top_block"], "border_block": t["border_block"],
               "film_sets": {a: [[int(L), int(f)] for L, f in s] for a, s in film_sets.items()
                             if a not in sets},
               "set_stats": {a: {"n_active": int(sum(act[li[L], f] > 0 for L, f in s)),
                                 "overlap_top": len(set(map(tuple, s)) & set(top))}
                             for a, s in film_sets.items()},
               "arms": {}}
        for arm in ARMS:
            edits = [] if arm == "base" else runner.edits(film_sets[arm], "scale", -1.0)
            rec["arms"][arm] = runner.answer(prompt, pil, edits, n_samples=args.n_samples,
                                             max_new_tokens=args.max_new_tokens)
        rows.append(rec)
        log(f"[{k + 1}/{len(tests)} {time.time() - t0:.0f}s] {g} {t['image']}  " + "  ".join(
            f"{a}: {v['norm']:.2f} s={yes_rate(v['samples_parsed']):.2f}"
            for a, v in rec["arms"].items()))
        prior["rows"] = rows
        args.out.write_text(json.dumps(prior, indent=1))

    res = analyze(rows)
    report(res)
    prior.update({"versions": {"torch": torch.__version__, "transformers": transformers_version},
                  "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens,
                  "seed": args.seed, **res})
    args.out.write_text(json.dumps(prior, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
