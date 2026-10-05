"""The pre-registered analysis of `scripts/clamp_contrastive_answers.py` (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "clamp_contrastive_answers.py"
_spec = importlib.util.spec_from_file_location("clamp_contrastive_answers", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _rec(n_yes, n=30):
    return {"raw": n_yes / n, "norm": 1.0, "samples_parsed": ["YES"] * n_yes + ["NO"] * (n - n_yes)}


def _rows(group, n, yes_by_arm, rng):
    rows = []
    for i in range(n):
        arms = {a: _rec(int(np.clip(y + rng.integers(-2, 3), 0, 30)))
                for a, y in yes_by_arm.items()}
        rows.append({"group": group, "image": f"{group}{i}", "arms": arms,
                     "set_stats": {a: {"n_active": 80, "overlap_top": 0} for a in mod.ARMS}})
    return rows


BASE = dict.fromkeys(mod.ARMS, 29)


def test_double_dissociation_is_detected() -> None:
    rng = np.random.default_rng(0)
    rows = (_rows("tp_cardiomegaly", 20, {**BASE, "con_card": 10, "con_eff": 28}, rng)
            + _rows("tp_effusion", 20, {**BASE, "con_card": 28, "con_eff": 10}, rng)
            + _rows("fp", 40, {**BASE, "con_card": 12, "anch_evid": 15}, rng))
    v = mod.analyze(rows, n_boot=500)["verdicts"]
    assert v["S1"] == "SELECTION RESCUES SPECIFICITY"
    assert v["S2"] == "CONTRAST MOVES THE FALSE POSITIVE"
    assert v["S3"] == "LOCATION MATTERS"


def test_inert_contrast_sets() -> None:
    rng = np.random.default_rng(1)
    arms = {**BASE, "top": 15}
    rows = (_rows("tp_cardiomegaly", 20, arms, rng) + _rows("tp_effusion", 20, arms, rng)
            + _rows("fp", 40, arms, rng))
    res = mod.analyze(rows, n_boot=500)
    v = res["verdicts"]
    assert v["S1"] == "CONTRASTIVE SETS INERT"
    assert v["S2"] == "CONTRAST INERT ON FP" and v["S3"] == "LOCATION DOES NOT"
    # the readout verdicts are computed too: P_norm stays at 1.0 while `top` drops the answer
    assert res["summary"]["fp"]["top"]["norm_verdict"] == "MISREADS"
