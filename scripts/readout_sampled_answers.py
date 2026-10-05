"""Sampled answers as the readout: how far is each first-token readout from what the model says?

`readout_generation_check.md` (E2) found that under feature suppression the yes/no-normalised
readout P_norm and MedGemma's generated answer disagree (DISAGREES in all three clamp arms). The
lost first-token probability goes to "Based", which opens a reasoned answer, and the clamp changes
what that reasoning concludes. E2 had 10 samples per film, 20 films chosen in August by a
raw-readout threshold, and one finding. This measures the quantity a reader of a yes/no evaluation
actually cares about, the probability that the model's written answer is yes, with enough samples
and films to put intervals on it, on films chosen by the answer itself, and on two more findings.

FILMS, drawn and fixed by this script before any arm runs. One film per patient; no patient in two
groups or among E2's films. Candidates are visited in one seeded permutation of the local NIH
images, the drawn groups taking turns; at most --max-scan eligible candidates per group.
  fp_banked        E2's 20 films (AP view, no Cardiomegaly label, the model said yes), with E2's
                   saved clamp feature sets, occlusion block and border block.
  fp_fresh         20 films drawn by the same population rule, now applied to the generated answer:
                   AP view, no "Cardiomegaly" in the NIH labels, and the plain model's greedy answer
                   to the cardiomegaly question parses YES.
  tp_cardiomegaly, tp_effusion, tp_pneumothorax
                   20 each: the NIH label is present and the greedy answer to that finding's
                   question parses YES.
Occlusion block of a drawn film: the block (8x8 grid of 2x2 image tokens) whose black fill lowers
the yes-minus-no logit gap most. E2's films keep their recorded block (chosen by the raw readout in
`attn_causal_map.py`). Border block: one block drawn uniformly from the grid's edge (seeded).

ARMS (same operators as E2; clamps are the film's own top-80 image features by mean activation,
at image positions; E2's films reuse E2's saved sets, so the selection boundary cannot move):
  fp_*   base, border, blur, mean, black, clamp_scale-1, clamp_ablate+0, clamp_scale-2
  tp_*   base, border, black, clamp_scale-1

RECORDED per film x arm: P_raw, P_norm, answer mass, logit gap, top-10 first tokens, the greedy
answer, and --n-samples answers at temperature 1 (top_k=0, top_p=1), max --max-new-tokens tokens,
each parsed with E2's rule and its opening word kept. Sampling is seeded per film.

PRE-SPECIFIED, fixed before running (commit = pre-registration).
s = fraction of a film's samples that parse YES: the probability that the written answer says yes.
For readout R in {P_raw, P_norm} and an arm, per film,
    diff = (R_base - R_arm) - (s_base - s_arm),
the readout's drop minus the drop in what the model says. With the film-bootstrap interval of
mean(diff) (10,000 resamples):
    MISREADS     interval excludes 0 and |mean(diff)| > 0.05
    TRACKS       interval inside [-0.05, 0.05]
    UNRESOLVED   otherwise
V1 headline, fp_banked + fp_fresh (n = 40), each clamp arm. "P_norm misreads the clamp's effect on
   the answer" is SUPPORTED if P_norm MISREADS in all three clamp arms, REFUTED if it TRACKS in all
   three, else PARTIAL. The same statement is evaluated for P_raw.
V2 replication of E2 on fp_fresh: E2's rule (`arm_verdict`, greedy answer vs the P_norm call) per
   clamp arm. REPLICATES if DISAGREES in at least 2 of 3 arms, else DOES NOT REPLICATE.
V3 other findings, tp_* under clamp_scale-1. A finding shows an EFFECT if the interval of the
   sampled drop (s_base - s_arm) excludes 0. GENERALISES if at least 2 findings show an EFFECT and
   P_norm MISREADS on every finding with an EFFECT; DOES NOT GENERALISE if P_norm TRACKS on some
   finding with an EFFECT; INCONCLUSIVE otherwise.
Reported, not verdicts: every verdict again with s replaced by YES/(YES+NO) (samples that answer);
each group separately; the occlusion arms; mean |R - s| per arm; and, pooled over samples, the share
opening with each word and P(YES | opening word).

GPU. Usage (VM):
  python scripts/readout_sampled_answers.py
  python scripts/readout_sampled_answers.py \
    --analyze-only results/clt_scale/readout_sampled_answers.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tracecxr.intervention.answer_runner import (
    PROMPTS,
    border_blocks,
    draw_films,
    fill,
    patient,
)
from tracecxr.intervention.generation_check import (
    arm_verdict,
    bootstrap_ci,
    call,
    effect_verdict,
    tally,
    yes_rate,
)

N_FEATS = 80
CLAMP_OPS = {"clamp_scale-1": ("scale", -1.0), "clamp_ablate+0": ("ablate", 0.0),
             "clamp_scale-2": ("scale", -2.0)}
FP_ARMS = ("base", "border", "blur", "mean", "black", *CLAMP_OPS)
TP_ARMS = ("base", "border", "black", "clamp_scale-1")
GROUPS = ("fp_banked", "fp_fresh", "tp_cardiomegaly", "tp_effusion", "tp_pneumothorax")
FINDING = {"fp_banked": "cardiomegaly", "fp_fresh": "cardiomegaly",
           "tp_cardiomegaly": "cardiomegaly", "tp_effusion": "effusion",
           "tp_pneumothorax": "pneumothorax"}
ELIGIBLE = {
    "fp_fresh": lambda lab, view: view == "AP" and "Cardiomegaly" not in lab,
    "tp_cardiomegaly": lambda lab, _v: "Cardiomegaly" in lab,
    "tp_effusion": lambda lab, _v: "Effusion" in lab,
    "tp_pneumothorax": lambda lab, _v: "Pneumothorax" in lab,
}
MARGIN = 0.05


def log(*a: object) -> None:
    print(*a, flush=True)


def arms_for(group: str) -> tuple[str, ...]:
    return FP_ARMS if group.startswith("fp_") else TP_ARMS


# ---------------------------------------------------------------------------------------------
# analysis (pure; runs on the saved JSON)
# ---------------------------------------------------------------------------------------------
def _rates(rec: dict, given_answer: bool) -> float:
    return yes_rate(rec["samples_parsed"], given_answer=given_answer)


def summarize_arm(rows: list[dict], arm: str, *, n_boot: int = 10000) -> dict:
    """Readouts, sampled rates, agreement and effect verdicts for one arm over ``rows``."""
    A = [r["arms"][arm] for r in rows]
    B = [r["arms"]["base"] for r in rows]
    raw = np.array([a["raw"] for a in A])
    norm = np.array([a["norm"] for a in A])
    greedy = [a["greedy_parsed"] for a in A]
    samp = tally(p for a in A for p in a["samples_parsed"])
    ns = max(sum(samp.values()), 1)
    out: dict = {
        "n": len(rows), "raw": float(raw.mean()), "norm": float(norm.mean()),
        "mass": float(np.mean([a["mass"] for a in A])),
        "gap_change_median": float(np.median([a["gap"] - b["gap"] for a, b in zip(A, B,
                                                                                   strict=True)])),
        "flips_raw": int(sum(call(p) == "NO" for p in raw)),
        "flips_norm": int(sum(call(p) == "NO" for p in norm)),
        "greedy": tally(greedy),
        "greedy_eq_raw": int(sum(call(p) == g for p, g in zip(raw, greedy, strict=True))),
        "greedy_eq_norm": int(sum(call(p) == g for p, g in zip(norm, greedy, strict=True))),
        "sampled": {k: v / ns for k, v in samp.items()},
    }
    for tag, given in (("s", False), ("c", True)):
        rate = np.array([_rates(a, given) for a in A])
        base = np.array([_rates(b, given) for b in B])
        out[f"{tag}_mean"] = bootstrap_ci(rate, n_boot=n_boot)
        out[f"abs_err_raw_{tag}"] = float(np.nanmean(np.abs(raw - rate)))
        out[f"abs_err_norm_{tag}"] = float(np.nanmean(np.abs(norm - rate)))
        if arm == "base":
            continue
        drop = base - rate
        out[f"{tag}_drop"] = bootstrap_ci(drop, n_boot=n_boot)
        for rname, R, R0 in (("raw", raw, np.array([b["raw"] for b in B])),
                             ("norm", norm, np.array([b["norm"] for b in B]))):
            diff = (R0 - R) - drop
            out[f"{rname}_drop"] = bootstrap_ci(R0 - R, n_boot=n_boot)
            out[f"{rname}_vs_{tag}"] = {"diff": bootstrap_ci(diff, n_boot=n_boot),
                                        "verdict": effect_verdict(diff, margin=MARGIN,
                                                                  n_boot=n_boot)}
    if arm in CLAMP_OPS:
        out["e2_rule"] = arm_verdict(list(norm), greedy)
    # decomposition by opening word, pooled over films and samples
    words = [w for a in A for w in a["samples_open"]]
    said = [p for a in A for p in a["samples_parsed"]]
    cnt = Counter(words)
    out["opening"] = {w: {"share": c / max(len(words), 1),
                          "p_yes": float(np.mean([p == "YES" for x, p in zip(words, said,
                                                                              strict=True)
                                                  if x == w]))}
                      for w, c in cnt.most_common(6)}
    first = Counter()
    for a in A:
        for _, t, p in a["top10"]:
            first[t.strip() or repr(t)] += p / len(A)
    out["first_token"] = dict(first.most_common(6))
    return out


def verdicts(summary: dict) -> dict:
    """V1-V3 from the per-set summaries (see the module docstring)."""
    v: dict = {}
    for tag in ("s", "c"):
        fp = summary["fp"]
        for rname in ("raw", "norm"):
            vs = [fp[a][f"{rname}_vs_{tag}"]["verdict"] for a in CLAMP_OPS]
            v[f"V1_{rname}_{tag}"] = ("SUPPORTED" if all(x == "MISREADS" for x in vs) else
                                      "REFUTED" if all(x == "TRACKS" for x in vs) else "PARTIAL")
            v[f"V1_{rname}_{tag}_arms"] = vs
        tp = {g: summary[g]["clamp_scale-1"] for g in GROUPS if g.startswith("tp_")
              and g in summary}
        effect = {g: (s[f"{tag}_drop"][1] > 0 or s[f"{tag}_drop"][2] < 0) for g, s in tp.items()}
        nv = {g: s[f"norm_vs_{tag}"]["verdict"] for g, s in tp.items()}
        with_eff = [g for g in tp if effect[g]]
        v[f"V3_{tag}"] = ("GENERALISES" if len(with_eff) >= 2
                          and all(nv[g] == "MISREADS" for g in with_eff) else
                          "DOES NOT GENERALISE" if any(nv[g] == "TRACKS" for g in with_eff) else
                          "INCONCLUSIVE")
        v[f"V3_{tag}_detail"] = {g: {"effect": effect[g], "norm": nv[g]} for g in tp}
    if "fp_fresh" in summary:
        e2 = [summary["fp_fresh"][a]["e2_rule"] for a in CLAMP_OPS]
        v["V2"] = "REPLICATES" if sum(x == "DISAGREES" for x in e2) >= 2 else "DOES NOT REPLICATE"
        v["V2_arms"] = e2
    return v


def analyze(rows: list[dict], *, n_boot: int = 10000) -> dict:
    sets = {"fp": [r for r in rows if r["group"].startswith("fp_")]}
    for g in GROUPS:
        sets[g] = [r for r in rows if r["group"] == g]
    summary = {name: {arm: summarize_arm(rs, arm, n_boot=n_boot)
                      for arm in arms_for(rs[0]["group"])}
               for name, rs in sets.items() if rs}
    return {"summary": summary, "verdicts": verdicts(summary)}


def report(res: dict) -> None:
    for name, arms in res["summary"].items():
        log(f"\n=== {name} (n={next(iter(arms.values()))['n']}) ===")
        log(f"  {'arm':15s}{'P_raw':>7s}{'P_norm':>7s}{'s':>7s}  {'s drop [95% CI]':>22s}"
            f"  {'raw-s':>15s}  {'norm-s':>15s}")
        for arm, a in arms.items():
            sd = a.get("s_drop")
            line = (f"  {arm:15s}{a['raw']:7.3f}{a['norm']:7.3f}{a['s_mean'][0]:7.3f}  "
                    + (f"{sd[0]:+.3f} [{sd[1]:+.3f},{sd[2]:+.3f}]" if sd else " " * 22))
            for r in ("raw", "norm"):
                if f"{r}_vs_s" in a:
                    line += f"  {a[f'{r}_vs_s']['diff'][0]:+.3f} {a[f'{r}_vs_s']['verdict']:>9s}"
            log(line)
    log("\n=== verdicts ===")
    for k, v in res["verdicts"].items():
        log(f"  {k}: {v}")


# ---------------------------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------------------------
def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/clt_scale"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--ckpt", default="/mnt/fast/clt/orig2048/clt_ckpt.pt")
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--max-scan", type=int, default=600)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--groups", default=",".join(GROUPS), help="subset, for debugging")
    ap.add_argument("--analyze-only", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("/mnt/fast/clt/readout_sampled_answers.json"))
    args = ap.parse_args()

    if args.analyze_only:
        d = json.loads(args.analyze_only.read_text())
        res = analyze(d["rows"])
        report(res)
        return

    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from tracecxr.intervention.answer_runner import AnswerRunner  # noqa: PLC0415
    from tracecxr.intervention.generation_check import parse_answer  # noqa: PLC0415
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    groups = [g for g in args.groups.split(",") if g in GROUPS]
    runner = AnswerRunner.load(args.results, args.ckpt)
    log(f"loaded MedGemma + CLT ({len(runner.layers)} layers); groups {groups}")
    img_dir = args.data_dir / "images"

    e2 = json.loads((args.results / "readout_generation_check.json").read_text())["rows"]
    used = {patient(r["image"]) for r in e2}
    prior = json.loads(args.out.read_text()) if args.out.exists() else None
    if prior and prior.get("draw"):
        draw = prior["draw"]
        log(f"resuming: draw and {len(prior['rows'])} finished films from {args.out}")
    else:
        df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
        have = {p.name for p in img_dir.glob("*.png") if not p.name.startswith("._")}
        df = df[df["Image Index"].isin(have)]
        pool = [(str(r["Image Index"]), str(r["Finding Labels"]), str(r["View Position"]))
                for _, r in df.iterrows()]
        order = np.random.default_rng(args.seed).permutation(len(pool))
        pool = [pool[i] for i in order]
        log(f"{len(pool)} local films; {len(used)} E2 patients excluded")
        greedy_log: dict[str, str] = {}

        def accept(g: str, name: str) -> bool:
            pil = Image.open(img_dir / name).convert("RGB")
            text = runner.plain_greedy(PROMPTS[FINDING[g]], pil, args.max_new_tokens)
            greedy_log[f"{g}/{name}"] = text
            return parse_answer(text) == "YES"

        drawn_groups = {g: ELIGIBLE[g] for g in groups if g in ELIGIBLE}
        picks = draw_films(pool, drawn_groups, accept, n=args.n_per_group, used=used,
                           max_scan=args.max_scan)
        brng = np.random.default_rng(args.seed + 1)
        edge = border_blocks()
        draw = {"fp_banked": [{"image": r["image"], "top_block": r["top_block"],
                               "border_block": r["border_block"],
                               "clamp_pairs": r["clamp_pairs"],
                               "boundary_gap": r["boundary_gap"]} for r in e2]
                if "fp_banked" in groups else []}
        for g, names in picks.items():
            draw[g] = []
            for name in names:
                arr = np.array(Image.open(img_dir / name).convert("RGB"))
                gaps = runner.scan_blocks(PROMPTS[FINDING[g]], arr)
                draw[g].append({"image": name, "top_block": int(np.argmin(gaps)),
                                "border_block": int(brng.choice(edge)), "block_gaps": gaps})
            n_scanned = sum(1 for k in greedy_log if k.startswith(g + "/"))
            log(f"  {g}: {len(names)} films accepted of {n_scanned} eligible candidates scanned")
        prior = {"draw": draw, "greedy_draw": greedy_log, "rows": []}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prior, indent=1))

    rows: list[dict] = list(prior["rows"])
    done = {(r["group"], r["image"]) for r in rows}
    t0 = time.time()
    films = [(g, f) for g in groups for f in draw.get(g, [])]
    for k, (g, film) in enumerate(films):
        name = film["image"]
        if (g, name) in done:
            continue
        torch.manual_seed(args.seed + 1000 * GROUPS.index(g) + k)
        prompt = PROMPTS[FINDING[g]]
        pil = Image.open(img_dir / name).convert("RGB")
        arr = np.array(pil)
        rec = {"group": g, "image": name, "finding": FINDING[g], **film, "arms": {}}
        if "clamp_pairs" not in rec:
            score = runner.image_feats(prompt, pil).mean(0)
            rec["clamp_pairs"] = [[int(L), int(f)] for L, f in runner.topn(score, N_FEATS)]
            srt = np.sort(score, axis=None)[::-1]
            rec["boundary_gap"] = float(srt[N_FEATS - 1] - srt[N_FEATS])
        for arm in arms_for(g):
            image, edits = pil, []
            if arm in ("black", "mean", "blur"):
                image = fill(arr, rec["top_block"], arm)
            elif arm == "border":
                image = fill(arr, rec["border_block"], "black")
            elif arm in CLAMP_OPS:
                edits = runner.edits(rec["clamp_pairs"], *CLAMP_OPS[arm])
            rec["arms"][arm] = runner.answer(prompt, image, edits, n_samples=args.n_samples,
                                             max_new_tokens=args.max_new_tokens)
        rows.append(rec)
        log(f"[{k + 1}/{len(films)} {time.time() - t0:.0f}s] {g} {name}  " + "  ".join(
            f"{a}: {v['raw']:.2f}/{v['norm']:.2f} s={yes_rate(v['samples_parsed']):.2f}"
            for a, v in rec["arms"].items()))
        prior["rows"] = rows
        args.out.write_text(json.dumps(prior, indent=1))

    res = analyze(rows)
    report(res)
    prior.update({"versions": {"torch": torch.__version__,
                               "transformers": transformers_version},
                  "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens,
                  "seed": args.seed, **res})
    args.out.write_text(json.dumps(prior, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
