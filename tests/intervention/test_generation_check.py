"""The parse rule and pre-registered verdict for `scripts/readout_generation_check.py`."""

from __future__ import annotations

import pytest
from tracecxr.intervention.generation_check import (
    arm_verdict,
    call,
    parse_answer,
    replay_border_blocks,
    tally,
)


@pytest.mark.parametrize(("text", "want"), [
    ("Yes", "YES"),
    ("**Yes**", "YES"),
    ("no.", "NO"),
    ("**No**, the heart size is normal.", "NO"),
    ("Based on the image, there is no evidence of cardiomegaly.", "NO"),
    ("Based on the image, the cardiac silhouette is enlarged. Yes.", "YES"),
    ("The heart is not enlarged.", "NONE"),  # "not" is not "no"
    ("I don't know", "NONE"),               # "know" contains "no" but is not the word
    ("## Answer\n\nYES", "YES"),
    ("", "NONE"),
])
def test_parse_answer(text: str, want: str) -> None:
    assert parse_answer(text) == want


def test_call_threshold_is_inclusive() -> None:
    assert call(0.5) == "YES" and call(0.4999) == "NO"


def test_tally_has_every_key() -> None:
    assert tally(["YES", "YES", "NONE"]) == {"YES": 2, "NO": 0, "NONE": 1}


def test_replay_border_blocks_is_on_the_border_and_deterministic() -> None:
    a, b = replay_border_blocks(20), replay_border_blocks(20)
    assert a == b and len(a) == 20
    for blk in a:
        r, c = divmod(blk, 8)
        assert r in (0, 7) or c in (0, 7)
    # a prefix replay matches: each film's block depends only on its position
    assert replay_border_blocks(5) == a[:5]


def test_verdict_validated() -> None:
    norm = [0.99] * 20
    assert arm_verdict(norm, ["YES"] * 19 + ["NONE"]) == "VALIDATED"


def test_verdict_disagrees_takes_priority() -> None:
    norm = [0.99] * 20
    greedy = ["NO"] * 3 + ["NONE"] * 6 + ["YES"] * 11
    assert arm_verdict(norm, greedy) == "DISAGREES"


def test_verdict_hedge_dominated() -> None:
    assert arm_verdict([0.99] * 20, ["NONE"] * 5 + ["YES"] * 15) == "HEDGE-DOMINATED"


def test_verdict_no_on_a_no_call_is_agreement_not_disagreement() -> None:
    # films the decision readout itself calls NO do not count against it
    norm = [0.1] * 4 + [0.99] * 16
    assert arm_verdict(norm, ["NO"] * 4 + ["YES"] * 16) == "VALIDATED"


def test_verdict_mixed() -> None:
    norm = [0.99] * 20
    assert arm_verdict(norm, ["NO"] * 2 + ["NONE"] * 2 + ["YES"] * 16) == "MIXED"


def test_verdict_length_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry per film"):
        arm_verdict([0.9], [])
