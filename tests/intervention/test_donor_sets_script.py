"""Pure helpers of `scripts/donor_sets_patient_disjoint.py` (the GPU body runs only on the VM)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "donor_sets_patient_disjoint.py"
_spec = importlib.util.spec_from_file_location("donor_sets_patient_disjoint", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_patient_id_parses_nih_names() -> None:
    assert mod.patient("00000013_006.png") == 13


def test_jaccard() -> None:
    assert mod.jaccard([(1, 2), (1, 3)], [(1, 3), (2, 2)]) == 1 / 3


def _rise(card, eff, ptx, rnd):
    return {"card_donor": card, "eff_donor": eff, "ptx_donor": ptx, "self": 0.0, "random": rnd}


def test_amp_verdict_generic() -> None:
    p = {"eff_donor": 0.8, "ptx_donor": 0.01}
    assert mod.amp_verdict(_rise(0.13, 0.12, 0.19, 0.01), p) == "GENERIC"


def test_amp_verdict_specific_needs_both_gap_and_p() -> None:
    assert mod.amp_verdict(_rise(0.40, 0.10, 0.10, 0.0),
                           {"eff_donor": 0.01, "ptx_donor": 0.01}) == "DONOR-SPECIFIC"
    assert mod.amp_verdict(_rise(0.40, 0.10, 0.10, 0.0),
                           {"eff_donor": 0.20, "ptx_donor": 0.01}) == "MIXED"


def test_amp_verdict_no_effect() -> None:
    assert mod.amp_verdict(_rise(0.05, 0.05, 0.05, 0.0), {"eff_donor": 1, "ptx_donor": 1}) \
        == "NO EFFECT"
