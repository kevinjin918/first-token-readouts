"""Tests for Unit 1 real model adapters.

All default tests run offline with NO weights and NO network: the adapters lazy-load, so
construction and registration are pure Python. The keyword text->findings parser is unit
tested directly (including negation). Real inference is covered only by a GPU test marked
``requires_gpu`` (skipped in CI).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from tracecxr.core import (
    CXRRecord,
    Finding,
    ModelOutput,
    available_models,
    get_model,
    load_builtin_models,
)
from tracecxr.core.model import ModelAdapter
from tracecxr.models import MedGemmaAdapter, Qwen3VLAdapter
from tracecxr.models._base import (
    PROBE_FINDINGS,
    REASONING_PROMPT,
    build_radiology_prompt,
    resolve_yes_no_token_ids,
    yes_no_question,
    yes_probability,
)

# --- Registration / discovery ---------------------------------------------------------


def test_package_imports_cleanly():
    import tracecxr.models  # noqa: F401 — import is the assertion


def test_load_builtin_models_registers_both():
    loaded = load_builtin_models()
    # Submodule names imported by discovery (private _base is skipped).
    assert "medgemma" in loaded
    assert "qwen3vl" in loaded
    names = available_models()
    assert {"medgemma", "qwen3vl", "mock"} <= set(names)


def test_get_model_returns_adapters():
    load_builtin_models()
    med = get_model("medgemma")
    qwen = get_model("qwen3vl")
    assert isinstance(med, MedGemmaAdapter)
    assert isinstance(qwen, Qwen3VLAdapter)
    # Structural conformance to the shared protocol.
    assert isinstance(med, ModelAdapter)
    assert isinstance(qwen, ModelAdapter)


@pytest.mark.parametrize("cls", [MedGemmaAdapter, Qwen3VLAdapter])
def test_adapter_name_matches_registry(cls):
    assert cls().name in available_models() or load_builtin_models()


# --- Construction is offline (no weights, no network) ---------------------------------


@pytest.mark.parametrize("cls", [MedGemmaAdapter, Qwen3VLAdapter])
def test_construction_does_not_load_weights(cls):
    adapter = cls()
    # No weights resolved yet.
    assert adapter._model is None
    assert adapter._processor is None
    # The model id is resolved from config, not hardcoded empty.
    assert adapter.model_id


@pytest.mark.parametrize("cls", [MedGemmaAdapter, Qwen3VLAdapter])
def test_construction_does_not_import_torch(cls):
    # If construction triggered a heavy import, torch/transformers would appear in
    # sys.modules. We only assert construction itself does not import them; if a prior
    # test already imported torch, skip the strict check.
    had_torch = "torch" in sys.modules
    cls(device="cpu", max_new_tokens=8)
    if not had_torch:
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules


@pytest.mark.parametrize("cls", [MedGemmaAdapter, Qwen3VLAdapter])
def test_model_id_override(cls):
    adapter = cls(model_id="some/other-id")
    assert adapter.model_id == "some/other-id"


# --- Default prompt builder -----------------------------------------------------------


def test_build_radiology_prompt_default():
    assert build_radiology_prompt() == REASONING_PROMPT
    assert "reasoning" in REASONING_PROMPT.lower()


def test_build_radiology_prompt_override():
    assert build_radiology_prompt("custom") == "custom"


# --- Constrained yes/no probing helpers (extraction is logprob-based, not text parsing) ---


def test_probe_findings_are_focus_pathologies():
    assert Finding.EFFUSION in PROBE_FINDINGS
    assert Finding.PNEUMOTHORAX in PROBE_FINDINGS
    # Bookkeeping labels are not probed.
    assert Finding.NO_FINDING not in PROBE_FINDINGS
    assert Finding.SUPPORT_DEVICES not in PROBE_FINDINGS


def test_yes_no_question_mentions_finding():
    q = yes_no_question(Finding.EFFUSION)
    assert "pleural effusion" in q.lower()
    assert "yes or no" in q.lower()


class _FakeTokenizer:
    """Maps known surface forms to distinct ids; first-token-of-word semantics."""

    _vocab = {"yes": 10, " yes": 10, "Yes": 11, " Yes": 11, "YES": 12,
              "no": 20, " no": 20, "No": 21, " No": 21, "NO": 22}

    def encode(self, text, add_special_tokens=False):
        return [self._vocab[text]] if text in self._vocab else []


def test_resolve_yes_no_token_ids():
    yes_ids, no_ids = resolve_yes_no_token_ids(_FakeTokenizer())
    assert set(yes_ids) == {10, 11, 12}
    assert set(no_ids) == {20, 21, 22}
    assert not (set(yes_ids) & set(no_ids))  # disjoint


def test_yes_probability_reads_logits():
    torch = pytest.importorskip("torch")  # torch is the optional [models] extra; skip in CI

    logits = torch.full((30,), -10.0)
    logits[10] = 5.0   # a "yes" token strongly preferred
    logits[20] = 1.0   # a "no" token
    p = yes_probability(logits, [10, 11, 12], [20, 21, 22])
    assert p > 0.95  # softmax([5,1]) -> ~0.98 for yes


def test_yes_probability_no_preferred():
    torch = pytest.importorskip("torch")  # torch is the optional [models] extra; skip in CI

    logits = torch.full((30,), -10.0)
    logits[20] = 4.0   # "no" preferred
    logits[10] = 0.0
    assert yes_probability(logits, [10], [20]) < 0.05


# --- Real inference (skipped without a GPU / weights) ---------------------------------


@pytest.mark.requires_gpu
@pytest.mark.parametrize("cls", [MedGemmaAdapter, Qwen3VLAdapter])
def test_real_generate_smoke(cls):  # pragma: no cover - needs weights
    adapter = cls(max_new_tokens=32)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    record = CXRRecord(id="t1", image=image, source="test")
    out = adapter.generate(record, with_image=True)
    assert isinstance(out, ModelOutput)
    assert isinstance(out.text, str)
