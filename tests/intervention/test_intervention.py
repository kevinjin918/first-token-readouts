"""Tests for the Stage 4 constrained-patching interface stub.

Covers the manifesto's "Stage 4: causal validation by intervention" contract: the
:class:`PatchSpec` dataclass (including negative steering over a layer range) is real, while
the heavy patching functions raise ``NotImplementedError`` until Stage 4 lands.
"""

from __future__ import annotations

import dataclasses

import pytest
from tracecxr.intervention import (
    PatchSpec,
    apply_patch,
    constrained_patch,
    measure_effect,
)


def test_patchspec_constructs_with_negative_steering_over_layer_range() -> None:
    """A negative-steering patch (factor=-1.0) over a layer range is the gold-standard suppress."""
    spec = PatchSpec(feature_id=42, factor=-1.0, layer_start=3, layer_end=7)

    assert spec.feature_id == 42
    assert spec.factor == -1.0
    assert spec.layer_start == 3
    assert spec.layer_end == 7


def test_patchspec_defaults_to_negative_steering() -> None:
    """Manifesto: steer negatively (multiply by -1.0) rather than zero-ablate."""
    spec = PatchSpec(feature_id=0)

    assert spec.factor == -1.0


def test_patchspec_supports_positive_activation_for_clean_case() -> None:
    """Other half of the bidirectional gold standard: activate to manufacture a hallucination."""
    spec = PatchSpec(feature_id=1, factor=2.5, layer_start=5, layer_end=9)

    assert spec.factor == 2.5


def test_patchspec_is_frozen() -> None:
    """A patch spec is an immutable record of one intervention."""
    spec = PatchSpec(feature_id=7)

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.factor = 0.0  # type: ignore[misc]


def test_apply_patch_is_stub() -> None:
    spec = PatchSpec(feature_id=0, factor=-1.0, layer_start=0, layer_end=2)
    with pytest.raises(NotImplementedError, match="Stage 4"):
        apply_patch(object(), spec)


def test_constrained_patch_is_stub() -> None:
    spec = PatchSpec(feature_id=0, factor=-1.0, layer_start=0, layer_end=2)
    with pytest.raises(NotImplementedError, match="Stage 4"):
        constrained_patch(object(), [spec])


def test_measure_effect_is_stub() -> None:
    with pytest.raises(NotImplementedError, match="Stage 4"):
        measure_effect(object(), object())
