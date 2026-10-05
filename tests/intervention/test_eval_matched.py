"""The pre-registered analysis of `scripts/readout_eval_matched.py` (E7, no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from tracecxr.intervention.generation_check import auc

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "readout_eval_matched.py"
_spec = importlib.util.spec_from_file_location("readout_eval_matched", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
CSV = Path.home() / ".cache/tracecxr/data/chestxray14/Data_Entry_2017.csv"


def test_pair_matrix_is_a_probability_of_ranking() -> None:
    p = np.array([0.1, 0.5, 0.9])
    m = mod.pair_matrix(p, p, 10)
    assert np.allclose(m + m.T, 1) and np.allclose(np.diag(m), 0.5)
    assert mod.pair_matrix(np.array([1.0]), np.array([0.0]), 5)[0, 0] == 1


def test_expected_auc_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    pp, pn = rng.beta(3, 2, 40), rng.beta(2, 3, 40)
    y = np.r_[np.ones(40), np.zeros(40)]
    mc = np.mean([auc(y, rng.binomial(10, np.r_[pp, pn])) for _ in range(4000)])
    assert abs(mod.pair_matrix(pp, pn, 10).mean() - mc) < 0.003


def test_weighted_counts_auc_equals_rank_auc_on_the_resample() -> None:
    rng = np.random.default_rng(1)
    k, y = rng.integers(0, 6, 30), np.arange(30) < 12
    i = rng.integers(0, 30, 30)
    w = np.bincount(i, minlength=30).astype(float)[None]
    got = mod.auc_counts(w[:, y], w[:, ~y], k[y], k[~y], 5)[0]
    assert np.isclose(got, auc(y[i], k[i]))


def _rows(n_films=150, n=50, follow=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for finding in ("cardiomegaly", "effusion", "pneumothorax"):
        for i in range(2 * n_films):
            lab = int(i < n_films)
            p = float(rng.beta(4, 2) if lab else rng.beta(2, 4))
            ps = p if follow else float(rng.uniform())      # answers that ignore the readout
            k = int(rng.binomial(n, ps))
            rows.append({"finding": finding, "image": f"{1336 + i:08d}_000.png", "label": lab,
                         "raw": p, "norm": p, "mass": 1.0,
                         "samples_parsed": ["YES"] * k + ["NO"] * (n - k)})
    return rows


def test_answers_drawn_from_the_readout_follow_it() -> None:
    res = mod.matched(_rows(), np.random.default_rng(2), n_boot=500)
    assert res["verdict"] == "ANSWERS FOLLOW THE READOUT"
    assert set(res["by_finding"]["effusion"]) >= {"auc_s", "raw", "norm"}


def test_answers_that_ignore_the_readout_depart_from_it() -> None:
    res = mod.matched(_rows(follow=False), np.random.default_rng(3), n_boot=500)
    assert res["verdict"] == "ANSWERS DEPART FROM THE READOUT"


def test_across_same_model_is_zero() -> None:
    d = {"model": "a", "n_samples": 50, "rows": _rows()}
    res = mod.across(d, d, np.random.default_rng(4), n_boot=200)["by_finding"]["cardiomegaly"]
    assert all(v == [0.0, 0.0, 0.0] for v in res.values())


def test_gate_refuses_wrong_films() -> None:
    rows = _rows(n_films=5)
    a = {"model": "a", "n_samples": 50, "rows": rows}
    sha = mod.digest(mod.films(rows))
    mod.gate(a, a, [], sha=sha)                                      # passes
    with pytest.raises(ValueError, match="E6"):
        mod.gate(a, a, [{"image": "00001336_004.png"}], sha=sha)
    with pytest.raises(ValueError, match="registered draw"):
        mod.gate(a, a, [], sha="0" * 64)
    old = {"model": "o", "n_samples": 50, "rows": [{**rows[0], "image": "00000013_006.png"}]}
    with pytest.raises(ValueError, match="images_002"):
        mod.gate(old, old, [], sha=sha)
    with pytest.raises(ValueError, match="samples"):
        mod.gate({**a, "n_samples": 20}, a, [], sha=sha)


@pytest.mark.skipif(not CSV.exists(), reason="NIH label CSV not downloaded")
def test_draw_reproduces_the_registered_films() -> None:
    fs = mod.films(mod.draw(CSV))
    assert len(fs) == 600 and mod.digest(fs) == mod.FILMS_SHA256
    assert all(mod.PATIENTS[0] <= int(f[1][:8]) <= mod.PATIENTS[1] for f in fs)
