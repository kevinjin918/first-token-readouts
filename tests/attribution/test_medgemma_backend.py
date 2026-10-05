"""CI-testable surface of the GPU attribution backend (manifesto §Stage 2).

The heavy ``trace`` path needs MedGemma + a CLT checkpoint on a GPU and is exercised only
on the H100. Here we cover what is verifiable offline: the pure output-token selection, the
protocol conformance, and that the module imports without torch.
"""

from __future__ import annotations

import numpy as np
import pytest
from tracecxr.attribution import AttributionBackend
from tracecxr.attribution.medgemma_backend import (
    MedGemmaCLTBackend,
    best_answer_token_id,
    select_output_token_ids,
)


class _MockTok:
    """A tiny tokenizer mapping casing/space variants to distinct ids."""

    _VOCAB = {"yes": 4443, " yes": 11262, "Yes": 10784, " Yes": 8438, "YES": 99}

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [self._VOCAB[s]] if s in self._VOCAB else []


def test_best_answer_token_id_picks_highest_scoring_variant() -> None:
    scores = np.zeros(20000)
    scores[10784] = 0.93  # "Yes" — what MedGemma actually emits
    scores[4443] = 0.0001  # "yes"
    assert best_answer_token_id(_MockTok(), "yes", scores) == 10784


def test_best_answer_token_id_handles_leading_space_variant() -> None:
    scores = np.zeros(20000)
    scores[11262] = 0.8  # " yes"
    assert best_answer_token_id(_MockTok(), "yes", scores) == 11262


def test_construction_is_lazy() -> None:
    # constructing the backend must not load the model/CLT (torch is pulled only in trace()).
    backend = MedGemmaCLTBackend("ckpt.pt", [8, 9, 10])
    assert backend._state == {}


def test_backend_satisfies_protocol() -> None:
    backend = MedGemmaCLTBackend("ckpt.pt", list(range(8, 20)))
    assert isinstance(backend, AttributionBackend)


def test_select_output_token_ids_covers_mass_then_stops() -> None:
    probs = np.zeros(100)
    probs[5], probs[9], probs[1] = 0.6, 0.3, 0.05
    chosen = select_output_token_ids(probs, n_logits=10, logit_mass=0.8)
    assert chosen == [5, 9]  # 0.6 + 0.3 >= 0.8, stop before the 0.05 tail


def test_select_output_token_ids_respects_n_logits_cap() -> None:
    probs = np.full(50, 0.02)
    chosen = select_output_token_ids(probs, n_logits=3, logit_mass=0.99)
    assert len(chosen) == 3


def test_select_output_token_ids_force_includes_finding() -> None:
    probs = np.zeros(100)
    probs[5], probs[9] = 0.7, 0.25
    chosen = select_output_token_ids(probs, finding_token_id=42, n_logits=10, logit_mass=0.9)
    assert 42 in chosen  # appended even though it is not in the top mass
    assert chosen[0] == 5


def test_select_output_token_ids_finding_already_present_not_duplicated() -> None:
    probs = np.zeros(100)
    probs[5], probs[9] = 0.7, 0.25
    chosen = select_output_token_ids(probs, finding_token_id=5, n_logits=10, logit_mass=0.9)
    assert chosen.count(5) == 1


@pytest.mark.requires_gpu
def test_trace_on_real_medgemma() -> None:  # pragma: no cover - GPU only
    pytest.skip("needs MedGemma weights + a CLT checkpoint on a GPU; run on the H100")
