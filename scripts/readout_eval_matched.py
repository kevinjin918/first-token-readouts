"""E7: without an intervention, do the written answers rank radiographs as the first-token readout
says they would, once the two scores have the same resolution? The corrected version of E6.

E6 (`readout_eval_answers.py`) compared AUC(P), P a continuous first-token readout, with AUC(s), s
the fraction of n sampled answers that say yes, which takes n + 1 values. A model that gives the
same answer on all n samples for most films (MedGemma, on 76 to 82% of E6's films) leaves s tied
where P still ranks, so AUC(P) - AUC(s) mixes resolution with any real disagreement between the
readout and the answers. That was a flaw in E6's design, found after its run
(`readout_eval_resolution.py`, post hoc). E7 is the corrected test, registered before its data
exist, on patients no earlier experiment has used.

MATCHED SCORE. If the written answers were independent draws that say yes with probability P, n of
them would give q = Binomial(n, P) / n, a score with s's resolution. Its expected AUC given the
films, E[AUC(q)], is computed exactly: for each positive film i and negative film j,
P(q_i > q_j) + P(q_i = q_j) / 2 from the two binomial distributions, averaged over the pairs. Then

    D = AUC(s) - E[AUC(q)]

is 0 in expectation if the answers follow the readout, and moves away from 0 if they rank films
differently from it, whatever the resolution. Ties count one half in every AUC.

FILMS. NIH ChestX-ray14 images_002: 10,000 films, 00001336_000 to 00003923_013 (patients 1336 to
3923), the CSV rows in that range (checked equal to the zip's contents). E2 to E6 drew only from
images_001 (patients 1 to 1335), and no file in `results/` names a patient in this range. E6's draw
(`draw_stratified`, unchanged), seed 20261008, 100 per class: per finding, 100 patients with the
NIH label and 100 without, one film per patient, 600 film-finding pairs (342 patients), the same
for both models. Hard gates before any analysis: every patient in [1336, 3923], none among E6's,
and the films equal that draw (FILMS_SHA256; `--check-draw` recomputes it from the CSV).

GENERATION. E6's script, unchanged, pointed at a directory that holds only images_002 and the label
CSV, with 50 samples instead of 20 and a new seed. Everything else is E6's: the paper's three
questions, T = 1 with top_k = 0, 128 new tokens, E2's parse rule, the first-token gate, CheXagent
per E6's amendments 1 and 2.
  python scripts/readout_eval_answers.py --model medgemma --data-dir DIR --n-samples 50 \
      --seed 20261008 --out results/clt_scale/readout_eval_matched_medgemma.json
  ~/venv-chexagent/bin/python scripts/readout_eval_answers.py --model chexagent --data-dir DIR \
      --n-samples 50 --seed 20261008 --out results/clt_scale/readout_eval_matched_chexagent.json

PRE-SPECIFIED, fixed before running (commit = pre-registration). Film bootstrap, 2,000 resamples,
films resampled jointly (P_raw and P_norm on the same resamples), seed 20261009, 95% percentile
intervals.
E7 per model, over the six D's (P = P_raw and P = P_norm, three findings):
    ANSWERS DEPART FROM THE READOUT   some |D| > 0.03 with an interval excluding 0
    ANSWERS FOLLOW THE READOUT        every interval inside [-0.03, 0.03]
    UNRESOLVED                        otherwise
This is E6's rule and margin, applied to D. Expected from E6's post-hoc analysis: FOLLOW in both
models.
Reported, not verdicts: per finding, AUC(P), AUC(s) and E[AUC(q)]; E6's own rule on the new films
(AUC(P) - AUC(s), printed and saved by E6's script); the share of films with s = 0 or s = 1;
AUC(P_raw) among films with s = 0; and across the two models, AUC_A - AUC_B under s and under
E[AUC(q)], with joint film-bootstrap intervals.

PLANNING, before any E7 data (simulation with E6's films as the population, 100 per class, s drawn
as Binomial(50, P), 200 runs of 1,000 resamples per finding, readout and model): the interval's
half-width is 0.006 to 0.016; it covers 0 in 85 to 96% of runs and lies inside [-0.03, 0.03] in 88
to 100%; a false DEPART occurs in at most 1%.

CPU. Usage:
  python scripts/readout_eval_matched.py A.json B.json --e6 E6.json --out OUT.json
  python scripts/readout_eval_matched.py --check-draw CSV
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom
from tracecxr.intervention.answer_runner import LABELS, patient
from tracecxr.intervention.generation_check import auc, eval_verdict

N_BOOT = 2000
SEED = 20261009
DRAW_SEED = 20261008
N_PER_CLASS = 100
N_SAMPLES = 50
PATIENTS = (1336, 3923)
FIRST, LAST = "00001336_000.png", "00003923_013.png"  # images_002; patient 3923 continues in 003
FILMS_SHA256 = "2297916e1a5c5f01f045b2b548fb73fec079fcfd98b97495404f5bfc70f19097"
READOUTS = ("raw", "norm")
VERDICT = {"READOUT CHANGES THE EVALUATION": "ANSWERS DEPART FROM THE READOUT",
           "INNOCUOUS": "ANSWERS FOLLOW THE READOUT", "UNRESOLVED": "UNRESOLVED"}


def log(*a: object) -> None:
    print(*a, flush=True)


def _e6():  # noqa: ANN202
    path = Path(__file__).resolve().parent / "readout_eval_answers.py"
    spec = importlib.util.spec_from_file_location("readout_eval_answers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def films(rows: list[dict]) -> list[tuple[str, str, int]]:
    return sorted((r["finding"], r["image"], int(r["label"])) for r in rows)


def digest(fs: list[tuple[str, str, int]]) -> str:
    return hashlib.sha256(json.dumps(sorted(fs)).encode()).hexdigest()


def draw(csv: Path) -> list[dict]:
    """E6's draw over the CSV rows from FIRST to LAST, which are exactly images_002's films."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv)
    df = df[(df["Image Index"] >= FIRST) & (df["Image Index"] <= LAST)]
    pool = [(str(r["Image Index"]), str(r["Finding Labels"])) for _, r in df.iterrows()]
    return _e6().draw_stratified(pool, N_PER_CLASS, DRAW_SEED)


def gate(a: dict, b: dict, e6_rows: list[dict], *, sha: str = FILMS_SHA256) -> None:
    """Refuse to analyse unless both runs answer exactly the registered fresh films."""
    for d in (a, b):
        if d["n_samples"] != N_SAMPLES:
            raise ValueError(f"{d['model']}: {d['n_samples']} samples, registered {N_SAMPLES}")
        if any(len(r["samples_parsed"]) != N_SAMPLES for r in d["rows"]):
            raise ValueError(f"{d['model']}: a row lacks {N_SAMPLES} parsed samples")
    fa = films(a["rows"])
    if fa != films(b["rows"]):
        raise ValueError("the two models answered different films")
    pats = {patient(f[1]) for f in fa}
    if not all(PATIENTS[0] <= p <= PATIENTS[1] for p in pats):
        raise ValueError("a patient outside images_002")
    if pats & {patient(r["image"]) for r in e6_rows}:
        raise ValueError("a patient shared with E6")
    if digest(fa) != sha:
        raise ValueError("films differ from the registered draw")


def yes_count(row: dict) -> int:
    return sum(p == "YES" for p in row["samples_parsed"])


def pair_matrix(pp: np.ndarray, pn: np.ndarray, n: int) -> np.ndarray:
    """``[i, j] = P(q_i > q_j) + P(q_i = q_j) / 2``, q ~ Binomial(n, p) / n; i pos., j neg."""
    k = np.arange(n + 1)
    fp = binom.pmf(k[None], n, np.asarray(pp)[:, None])
    fn = binom.pmf(k[None], n, np.asarray(pn)[:, None])
    below = np.cumsum(fn, 1) - fn
    return fp @ (below + 0.5 * fn).T


def auc_counts(wp: np.ndarray, wn: np.ndarray, kp: np.ndarray, kn: np.ndarray, n: int
               ) -> np.ndarray:
    """AUC of integer scores in 0..n, one value per row of film weights ``wp`` / ``wn``."""
    hp = np.stack([np.bincount(kp, w, n + 1) for w in wp])
    hn = np.stack([np.bincount(kn, w, n + 1) for w in wn])
    below = np.cumsum(hn, 1) - hn
    return (hp * (below + 0.5 * hn)).sum(1) / (hp.sum(1) * hn.sum(1))


def weights(n_films: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.bincount(rng.integers(0, n_films, n_films), minlength=n_films)
                     for _ in range(n_boot)]).astype(float)


def _ci(point: float, reps: np.ndarray) -> list[float]:
    return [float(point), *map(float, np.quantile(reps, [0.025, 0.975]))]


def matched(rows: list[dict], rng: np.random.Generator, *, n: int = N_SAMPLES,
            n_boot: int = N_BOOT) -> dict:
    """E7 for one model: D = AUC(s) - E[AUC(q)] per finding and readout, and the verdict."""
    out, ds = {}, []
    for finding in LABELS:
        rs = [r for r in rows if r["finding"] == finding]
        if not rs:
            continue
        y = np.array([r["label"] for r in rs]) == 1
        k = np.array([yes_count(r) for r in rs])
        w = weights(len(rs), n_boot, rng)
        wp, wn = w[:, y], w[:, ~y]
        keep = (wp.sum(1) > 0) & (wn.sum(1) > 0)
        wp, wn = wp[keep], wn[keep]
        s_pt = float(auc(y, k))
        s_rep = auc_counts(wp, wn, k[y], k[~y], n)
        res: dict = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "auc_s": s_pt}
        for key in READOUTS:
            p = np.array([r[key] for r in rs])
            m = pair_matrix(p[y], p[~y], n)
            q_pt = float(m.mean())
            q_rep = ((wp @ m) * wn).sum(1) / (wp.sum(1) * wn.sum(1))
            d = _ci(s_pt - q_pt, s_rep - q_rep)
            res[key] = {"auc_P": float(auc(y, p)), "eauc_q": q_pt, "D": d}
            ds.append(d)
        out[finding] = res
    return {"by_finding": out, "verdict": VERDICT[eval_verdict(ds)]}


def describe(d: dict) -> dict:
    """Reported, not verdicts: ties in s, ranking among s = 0 films, E6's rule as E6 saved it."""
    n = d["n_samples"]
    out = {}
    for finding in LABELS:
        rs = [r for r in d["rows"] if r["finding"] == finding]
        if not rs:
            continue
        y = np.array([r["label"] for r in rs])
        k = np.array([yes_count(r) for r in rs])
        raw = np.array([r["raw"] for r in rs])
        z = k == 0
        e6 = d.get("by_finding", {}).get(finding, {})
        out[finding] = {
            "tied": float(np.mean((k == 0) | (k == n))),
            "s0_n": int(z.sum()), "s0_pos": int(y[z].sum()),
            "s0_auc_raw": float(auc(y[z], raw[z])) if len(set(y[z])) == 2 else None,
            "e6_rule_diff": e6.get("diff"),
            "mass_mean": float(np.mean([r["mass"] for r in rs])),
        }
    return {"by_finding": out, "e6_rule_verdict": d.get("verdict")}


def across(a: dict, b: dict, rng: np.random.Generator, *, n: int = N_SAMPLES,
           n_boot: int = N_BOOT) -> dict:
    """Reported: AUC_A - AUC_B under s and under E[AUC(q)], films resampled jointly."""
    bm = {(r["finding"], r["image"]): r for r in b["rows"]}
    out = {}
    for finding in LABELS:
        ra = [r for r in a["rows"] if r["finding"] == finding]
        if not ra:
            continue
        rb = [bm[(finding, r["image"])] for r in ra]
        y = np.array([r["label"] for r in ra]) == 1
        w = weights(len(ra), n_boot, rng)
        wp, wn = w[:, y], w[:, ~y]
        keep = (wp.sum(1) > 0) & (wn.sum(1) > 0)
        wp, wn = wp[keep], wn[keep]
        ka, kb = (np.array([yes_count(r) for r in rr]) for rr in (ra, rb))
        rep_s = auc_counts(wp, wn, ka[y], ka[~y], n) - auc_counts(wp, wn, kb[y], kb[~y], n)
        res = {"s": _ci(auc(y, ka) - auc(y, kb), rep_s)}
        for key in READOUTS:
            pa, pb = (np.array([r[key] for r in rr]) for rr in (ra, rb))
            ma, mb = pair_matrix(pa[y], pa[~y], n), pair_matrix(pb[y], pb[~y], n)
            rep = (((wp @ ma) * wn).sum(1) - ((wp @ mb) * wn).sum(1)) / (wp.sum(1) * wn.sum(1))
            res[f"q{key}"] = _ci(float(ma.mean() - mb.mean()), rep)
        out[finding] = res
    return {"a": a["model"], "b": b["model"], "by_finding": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--e6", type=Path,
                    default=Path("results/clt_scale/readout_eval_answers_medgemma.json"))
    ap.add_argument("--out", type=Path, default=Path("results/clt_scale/readout_eval_matched.json"))
    ap.add_argument("--check-draw", type=Path, default=None, metavar="CSV")
    args = ap.parse_args()

    if args.check_draw:
        fs = films(draw(args.check_draw))
        log(f"{len(fs)} pairs, {len({patient(f[1]) for f in fs})} patients, sha256 {digest(fs)}"
            f" ({'matches' if digest(fs) == FILMS_SHA256 else 'DIFFERS from'} the registered draw)")
        return
    a, b = (json.loads(p.read_text()) for p in args.runs)
    gate(a, b, json.loads(args.e6.read_text())["rows"])
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "n_boot": N_BOOT, "n_samples": N_SAMPLES, "films_sha256": FILMS_SHA256,
           "models": {d["model"]: {"e7": matched(d["rows"], rng), "reported": describe(d)}
                      for d in (a, b)}}
    res["across"] = across(a, b, rng)
    args.out.write_text(json.dumps(res, indent=1))
    for m, r in res["models"].items():
        log(f"\n{m}")
        for f, v in r["e7"]["by_finding"].items():
            t = r["reported"]["by_finding"][f]
            log(f"  {f:13s} AUC(s) {v['auc_s']:.3f}  tied {t['tied']:.2f}  s=0 {t['s0_n']} "
                f"({t['s0_pos']} pos), AUC(P_raw) there {t['s0_auc_raw']}")
            for key in READOUTS:
                x = v[key]
                log(f"    {key:4s} AUC(P) {x['auc_P']:.3f}  E[AUC(q)] {x['eauc_q']:.3f}  "
                    f"D {x['D'][0]:+.3f} [{x['D'][1]:+.3f}, {x['D'][2]:+.3f}]")
        log(f"  >>> E7 ({m}): {r['e7']['verdict']}   (E6's rule on these films: "
            f"{r['reported']['e6_rule_verdict']})")
    log(f"\n{res['across']['a']} - {res['across']['b']}")
    for f, v in res["across"]["by_finding"].items():
        log(f"  {f:13s} " + "  ".join(f"{k} {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
                                      for k, x in v.items()))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
