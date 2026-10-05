"""Pure helpers and the pre-registered analysis of the sampled-answer experiments (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from tracecxr.intervention.answer_runner import (
    block_patch_ids,
    border_blocks,
    draw_films,
    fill,
    patient,
)
from tracecxr.intervention.generation_check import (
    auc,
    auc_bootstrap,
    bootstrap_ci,
    effect_verdict,
    eval_verdict,
    opening_word,
    yes_rate,
)

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "readout_sampled_answers.py"
_spec = importlib.util.spec_from_file_location("readout_sampled_answers", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# ---- generation_check helpers -----------------------------------------------------------------
@pytest.mark.parametrize(("text", "want"), [
    ("**Yes**, the heart is enlarged.", "yes"),
    ("Based on the image, no.", "based"),
    ("  The cardiac silhouette", "the"),
    ("123", ""),
])
def test_opening_word(text: str, want: str) -> None:
    assert opening_word(text) == want


def test_yes_rate_both_denominators() -> None:
    p = ["YES", "NO", "NONE", "YES"]
    assert yes_rate(p) == 0.5
    assert yes_rate(p, given_answer=True) == 2 / 3
    assert np.isnan(yes_rate(["NONE"], given_answer=True))


def test_bootstrap_ci_brackets_the_mean_and_drops_nan() -> None:
    m, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, float("nan")], n_boot=2000)
    assert m == pytest.approx(0.2) and lo <= m <= hi
    assert bootstrap_ci([5.0, 5.0], n_boot=100) == (5.0, 5.0, 5.0)


def test_effect_verdict() -> None:
    rng = np.random.default_rng(0)
    assert effect_verdict(0.25 + rng.normal(0, 0.05, 40), n_boot=2000) == "MISREADS"
    assert effect_verdict(rng.normal(0, 0.02, 40), n_boot=2000) == "TRACKS"
    assert effect_verdict(rng.normal(0.03, 0.4, 10), n_boot=2000) == "UNRESOLVED"
    # significant but immaterial is not a misread
    assert effect_verdict(np.full(40, 0.03), n_boot=200) == "TRACKS"


def test_auc_matches_hand_cases() -> None:
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auc([1, 1], [0.1, 0.2]))


def test_auc_bootstrap_is_paired() -> None:
    y = np.array([0] * 30 + [1] * 30)
    s = np.r_[np.linspace(0, 0.6, 30), np.linspace(0.4, 1, 30)]
    out = auc_bootstrap(y, {"a": s, "b": s}, n_boot=300)
    est, lo, hi = out["diff"]["a-b"]
    assert est == 0 and lo == 0 and hi == 0   # identical scores: paired difference is exactly 0


def test_eval_verdict() -> None:
    assert eval_verdict([(0.05, 0.02, 0.08)]) == "READOUT CHANGES THE EVALUATION"
    assert eval_verdict([(0.01, -0.02, 0.02), (0.0, -0.01, 0.01)]) == "INNOCUOUS"
    assert eval_verdict([(0.01, -0.05, 0.06)]) == "UNRESOLVED"


# ---- answer_runner pure helpers ---------------------------------------------------------------
def test_block_geometry() -> None:
    assert block_patch_ids(0) == [0, 1, 16, 17]
    assert block_patch_ids(63) == [238, 239, 254, 255]
    e = border_blocks()
    assert len(e) == 28 and 0 in e and 27 not in e and 63 in e


def test_fill_only_touches_the_block() -> None:
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = np.array(fill(arr, 9, "black"))     # block (1, 1): rows/cols 8..15
    assert out[8:16, 8:16].max() == 0 and out[:8].min() == 200 and out[16:].min() == 200
    with pytest.raises(ValueError):
        fill(arr, 0, "pink")


def test_draw_films_is_patient_disjoint_and_takes_turns() -> None:
    pool = [(f"{p:08d}_{k:03d}.png", lab, "AP")
            for p, lab in [(1, "A"), (2, "A|B"), (3, "B"), (4, "A"), (5, "B"), (6, "B")]
            for k in range(2)]
    seen = []

    def accept(g, name):
        seen.append((g, name))
        return patient(name) != 4          # the model says no on patient 4

    used = {6}
    picks = draw_films(pool, {"a": lambda lab, _v: "A" in lab, "b": lambda lab, _v: "B" in lab},
                       accept, n=2, used=used, max_scan=10)
    flat = [patient(x) for v in picks.values() for x in v]
    assert len(flat) == len(set(flat))     # one film per patient, no patient in two groups
    assert 6 not in flat and 4 not in flat
    assert picks["a"] == ["00000001_000.png"]          # patient 2 went to b, which drew next
    assert picks["b"] == ["00000002_000.png", "00000003_000.png"]
    assert used >= {1, 2, 3, 6}


# ---- the pre-registered analysis --------------------------------------------------------------
def _arm(raw, norm, n_yes, n_no=None, n=50, greedy="YES", first="yes"):
    n_no = n - n_yes if n_no is None else n_no
    parsed = ["YES"] * n_yes + ["NO"] * n_no + ["NONE"] * (n - n_yes - n_no)
    return {"raw": raw, "norm": norm, "mass": raw, "gap": 10 * norm,
            "top10": [[1, "Yes", raw], [2, "Based", 1 - raw]],
            "greedy_parsed": greedy, "samples_parsed": parsed,
            "samples_open": [first if p == "YES" else "based" for p in parsed]}


def _rows(group, n, rng, clamp_yes, spread=5):
    rows = []
    for i in range(n):
        arms = {"base": _arm(0.92, 1.0, 49)}
        for a in mod.FP_ARMS[1:] if group.startswith("fp_") else mod.TP_ARMS[1:]:
            if a.startswith("clamp"):
                k = (clamp_yes if spread == 0 else
                     min(int(rng.integers(clamp_yes - spread, clamp_yes + spread)), 50))
                arms[a] = _arm(0.62, 0.997, k, greedy="NO" if i < 4 else "YES")
            else:
                arms[a] = _arm(0.9, 0.99, 48)
        rows.append({"group": group, "image": f"{i:08d}_000.png", "arms": arms})
    return rows


def test_analysis_flags_norm_when_the_answer_moves_and_norm_does_not() -> None:
    rng = np.random.default_rng(1)
    rows = (_rows("fp_banked", 20, rng, 35) + _rows("fp_fresh", 20, rng, 35)
            + _rows("tp_cardiomegaly", 20, rng, 35) + _rows("tp_effusion", 20, rng, 35)
            + _rows("tp_pneumothorax", 20, rng, 49, spread=0))
    res = mod.analyze(rows, n_boot=500)
    v = res["verdicts"]
    assert v["V1_norm_s"] == "SUPPORTED"     # norm drop ~0.003 vs sampled drop ~0.28
    assert v["V2"] == "REPLICATES"           # greedy NO on 4 films where P_norm says yes
    assert v["V3_s"] == "GENERALISES"        # ptx has no effect, the other two misread
    assert v["V3_s_detail"]["tp_pneumothorax"]["effect"] is False
    s = res["summary"]["fp"]["clamp_scale-1"]
    assert s["flips_norm"] == 0 and s["opening"]["based"]["p_yes"] == 0.0


def test_analysis_refutes_when_norm_tracks() -> None:
    rng = np.random.default_rng(2)
    rows = _rows("fp_banked", 20, rng, 49) + _rows("fp_fresh", 20, rng, 49)
    for r in rows:
        for a in mod.CLAMP_OPS:
            r["arms"][a] = _arm(0.62, 0.997, 49)     # the answer does not move either
    v = mod.analyze(rows, n_boot=500)["verdicts"]
    assert v["V1_norm_s"] == "REFUTED" and v["V2"] == "DOES NOT REPLICATE"
