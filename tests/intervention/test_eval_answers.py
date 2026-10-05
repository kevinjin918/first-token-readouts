"""Film draw and the pre-registered analysis of `scripts/readout_eval_answers.py` (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "readout_eval_answers.py"
_spec = importlib.util.spec_from_file_location("readout_eval_answers", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_draw_is_stratified_and_one_film_per_patient() -> None:
    pool = [(f"{p:08d}_{k:03d}.png", "Cardiomegaly" if p % 3 == 0 else "No Finding")
            for p in range(60) for k in range(3)]
    films = mod.draw_stratified(pool, 100, seed=0)
    card = [f for f in films if f["finding"] == "cardiomegaly"]
    pos = [f for f in card if f["label"]]
    assert len(pos) == 20 and len(card) == 40          # capped by the 20 positive patients
    pats = [int(f["image"][:8]) for f in card]
    assert len(set(pats)) == len(pats)
    assert [f for f in films if f["finding"] == "effusion"] == []   # no positives, no negatives


def _rows(model_shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for finding in ("cardiomegaly", "effusion"):
        for i in range(120):
            lab = int(i < 60)
            s = float(np.clip(0.2 + 0.6 * lab + rng.normal(0, 0.2), 0, 1))
            raw = float(np.clip(s + model_shift + rng.normal(0, 0.05), 0, 1))
            rows.append({"finding": finding, "image": f"{i:08d}_000.png", "label": lab,
                         "raw": raw, "norm": raw, "s": s, "greedy": float(s >= 0.5),
                         "greedy_parsed": "YES" if s >= 0.5 else "NO", "mass": 0.9,
                         "samples_open": ["yes" if s >= 0.5 else "based"]})
    return rows


def test_analysis_runs_and_gives_a_verdict() -> None:
    res = mod.analyze(_rows(), n_boot=300)
    assert res["verdict"] in {"INNOCUOUS", "UNRESOLVED", "READOUT CHANGES THE EVALUATION"}
    card = res["by_finding"]["cardiomegaly"]
    assert card["n_pos"] == 60 and set(card["auc"]) == set(mod.SCORES)
    assert set(card["opening_by_label"]) == {"0", "1"}


def test_readout_that_ignores_the_label_changes_the_evaluation() -> None:
    rows = _rows()
    rng = np.random.default_rng(3)
    for r in rows:
        r["norm"] = float(rng.uniform())           # a readout unrelated to the answer
    assert mod.analyze(rows, n_boot=300)["verdict"] == "READOUT CHANGES THE EVALUATION"


def test_compare_same_model_is_same_ranking() -> None:
    a = {"model": "a", "rows": _rows()}
    assert mod.compare(a, a, n_boot=200)["verdict"] == "SAME RANKING"
