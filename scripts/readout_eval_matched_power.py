"""Planning for E7 (`readout_eval_matched.py`), run before any E7 data existed.

E6's films are the population. Under the hypothesis that the written answers follow the readout,
s is drawn as Binomial(n, P) / n; each simulated run resamples N positive and N negative films,
computes D = AUC(s) - E[AUC(q)] and its film-bootstrap interval exactly as E7 does, and records
the interval's half-width, whether it covers 0, whether it lies inside [-0.03, 0.03], and whether
E7's DEPART condition fires. The docstring of `readout_eval_matched.py` quotes the N = 100, n = 50
rows of `python scripts/readout_eval_matched_power.py 200 1000` (seed 2).

CPU, about 10 minutes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parent / "readout_eval_matched.py"
_spec = importlib.util.spec_from_file_location("readout_eval_matched", _PATH)
e7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e7)


def delta_ci(sp, sn, pp, pn, n, rng, n_boot):  # noqa: ANN001, ANN201
    m = e7.pair_matrix(pp, pn, n)
    n_pos, n_neg = len(sp), len(sn)
    w = e7.weights(n_pos + n_neg, n_boot, rng)
    wp, wn = w[:, :n_pos], w[:, n_pos:]
    ok = (wp.sum(1) > 0) & (wn.sum(1) > 0)
    wp, wn = wp[ok], wn[ok]
    one_p, one_n = np.ones((1, n_pos)), np.ones((1, n_neg))
    pt = e7.auc_counts(one_p, one_n, sp, sn, n)[0] - m.mean()
    eq = ((wp @ m) * wn).sum(1) / (wp.sum(1) * wn.sum(1))
    lo, hi = np.quantile(e7.auc_counts(wp, wn, sp, sn, n) - eq, [0.025, 0.975])
    return pt, lo, hi


def main() -> None:
    runs, n_boot = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(2)
    for model in ("medgemma", "chexagent"):
        d = json.loads(Path(f"results/clt_scale/readout_eval_answers_{model}.json").read_text())
        for f in ("cardiomegaly", "effusion", "pneumothorax"):
            rows = [r for r in d["rows"] if r["finding"] == f]
            for key in ("raw", "norm"):
                pos = np.array([r[key] for r in rows if r["label"] == 1])
                neg = np.array([r[key] for r in rows if r["label"] == 0])
                for big_n in (100, 117):
                    for n in (20, 50):
                        hw, cover, inside, depart = [], 0, 0, 0
                        for _ in range(runs):
                            pp, pn = rng.choice(pos, big_n), rng.choice(neg, big_n)
                            sp, sn = rng.binomial(n, pp), rng.binomial(n, pn)
                            pt, lo, hi = delta_ci(sp, sn, pp, pn, n, rng, n_boot)
                            hw.append((hi - lo) / 2)
                            cover += lo <= 0 <= hi
                            inside += (lo >= -0.03) and (hi <= 0.03)
                            depart += abs(pt) > 0.03 and (lo > 0 or hi < 0)
                        print(f"{model:9s} {f:12s} {key:4s} N={big_n} n={n}: hw {np.mean(hw):.3f} "
                              f"cover0 {cover / runs:.2f} inside {inside / runs:.2f} "
                              f"falseDEPART {depart / runs:.2f}", flush=True)


if __name__ == "__main__":
    main()
