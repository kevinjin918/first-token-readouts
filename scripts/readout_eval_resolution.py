"""E6 follow-up, NOT pre-registered (written after E6's verdicts): matched resolution.

E6 (`readout_eval_answers.py`) compares the AUC of a first-token readout (continuous) with that of
s, the fraction of n sampled answers that say yes (n + 1 levels). When the model gives the same
answer on all n samples for most films, s ties them and cannot rank them, while the readout still
can. Part of AUC(readout) - AUC(s) is then resolution, not a difference in what the model says.

Here the readout is put on s's footing: q = Binomial(n, P) / n, n draws of a yes/no answer with the
readout's probability, redrawn in every bootstrap replicate. If the written answers rank films as
n draws from the first-token readout would, AUC(s) - AUC(q) is about 0.

Per model and finding: AUC(s) - AUC(q) for P = P_raw and P = P_norm (paired film bootstrap, the
binomial redrawn per replicate; point estimate averaged over draws), the share of films at s = 0
or s = 1, and AUC(P_raw) among films with s = 0. Across the two models: AUC_A - AUC_B under q
(P_raw) and under s, films resampled jointly, as E6-rank does.

CPU. Usage:
  python scripts/readout_eval_resolution.py results/clt_scale/readout_eval_answers_medgemma.json \
      results/clt_scale/readout_eval_answers_chexagent.json \
      --out results/clt_scale/readout_eval_resolution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FINDINGS = ("cardiomegaly", "effusion", "pneumothorax")
N_BOOT = 2000
N_POINT = 500


def yes_frac(row: dict, n: int) -> float:
    return sum(p == "YES" for p in row["samples_parsed"]) / n


def ci(xs: list[float], point: float) -> list[float]:
    lo, hi = np.percentile(xs, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


def within(d: dict, rng: np.random.Generator) -> dict:
    n = d["n_samples"]
    out = {}
    for f in FINDINGS:
        rows = [r for r in d["rows"] if r["finding"] == f]
        y = np.array([r["label"] for r in rows])
        s = np.array([yes_frac(r, n) for r in rows])
        res = {"n": len(rows), "tied": float(np.mean((s == 0) | (s == 1)))}
        for k in ("raw", "norm"):
            p = np.array([r[k] for r in rows])
            point = np.mean([roc_auc_score(y, s) - roc_auc_score(y, rng.binomial(n, p) / n)
                             for _ in range(N_POINT)])
            boot = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                if len(set(y[i])) < 2:
                    continue
                boot.append(roc_auc_score(y[i], s[i])
                            - roc_auc_score(y[i], rng.binomial(n, p[i]) / n))
            res[f"s-q{k}"] = ci(boot, point)
        z = s == 0
        p = np.array([r["raw"] for r in rows])
        res["s0_n"], res["s0_pos"] = int(z.sum()), int(y[z].sum())
        res["s0_auc_raw"] = float(roc_auc_score(y[z], p[z])) if len(set(y[z])) == 2 else None
        out[f] = res
    return out


def across(a: dict, b: dict, rng: np.random.Generator) -> dict:
    n = a["n_samples"]
    assert b["n_samples"] == n
    bm = {(r["finding"], r["image"]): r for r in b["rows"]}
    out = {}
    for f in FINDINGS:
        ra = [r for r in a["rows"] if r["finding"] == f]
        rb = [bm[(r["finding"], r["image"])] for r in ra]
        assert all(x["label"] == z["label"] for x, z in zip(ra, rb, strict=True))
        y = np.array([r["label"] for r in ra])
        sa, sb = (np.array([yes_frac(r, n) for r in rr]) for rr in (ra, rb))
        pa, pb = (np.array([r["raw"] for r in rr]) for rr in (ra, rb))
        dq, ds = [], []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < 2:
                continue
            dq.append(roc_auc_score(y[i], rng.binomial(n, pa[i]) / n)
                      - roc_auc_score(y[i], rng.binomial(n, pb[i]) / n))
            ds.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        out[f] = {"qraw": ci(dq, float(np.mean(dq))),
                  "s": ci(ds, roc_auc_score(y, sa) - roc_auc_score(y, sb))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()
    a, b = (json.loads(p.read_text()) for p in (args.a, args.b))
    rng = np.random.default_rng(args.seed)
    res = {"note": "post hoc, not pre-registered", "seed": args.seed, "n_boot": N_BOOT,
           "models": {a["model"]: within(a, rng), b["model"]: within(b, rng)},
           "across": {"a": a["model"], "b": b["model"], "by_finding": across(a, b, rng)}}
    args.out.write_text(json.dumps(res, indent=1))
    for m, r in res["models"].items():
        print(m)
        for f, v in r.items():
            print(f"  {f:13s} s-q(raw) {v['s-qraw'][0]:+.3f} [{v['s-qraw'][1]:+.3f}, "
                  f"{v['s-qraw'][2]:+.3f}]  s-q(norm) {v['s-qnorm'][0]:+.3f} "
                  f"[{v['s-qnorm'][1]:+.3f}, {v['s-qnorm'][2]:+.3f}]  tied {v['tied']:.2f}"
                  f"  s=0 {v['s0_n']} ({v['s0_pos']} pos) AUC(P_raw) {v['s0_auc_raw']}")
    print(f"{a['model']} - {b['model']}")
    for f, v in res["across"]["by_finding"].items():
        print(f"  {f:13s} q(raw) {v['qraw'][0]:+.3f} [{v['qraw'][1]:+.3f}, {v['qraw'][2]:+.3f}]"
              f"  s {v['s'][0]:+.3f} [{v['s'][1]:+.3f}, {v['s'][2]:+.3f}]")


if __name__ == "__main__":
    main()
