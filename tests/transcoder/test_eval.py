"""Tests for transcoder reconstruction-fidelity metrics (FVU / L0).

Pure NumPy — runs in CI with no torch. Covers the FVU regimes that carry the A1/A3 meaning:
perfect=0, mean-baseline=1, worse-than-mean>1 (the image-token OOD signature).
"""

from __future__ import annotations

import numpy as np
from tracecxr.transcoder.eval import fvu, fvu_by_kind, l0, summarize


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((100, 8))


def test_fvu_perfect_is_zero():
    t = _data()
    assert fvu(t, t.copy()) == 0.0


def test_fvu_mean_baseline_is_one():
    t = _data()
    recon = np.broadcast_to(t.mean(axis=0, keepdims=True), t.shape)
    assert abs(fvu(t, recon) - 1.0) < 1e-9


def test_fvu_worse_than_mean_exceeds_one():
    """A constant far from the data reconstructs worse than the mean -> FVU > 1 (the A1 case)."""
    t = _data()
    recon = np.full_like(t, 50.0)
    assert fvu(t, recon) > 1.0


def test_fvu_mask_selects_positions():
    t = _data()
    recon = t.copy()
    recon[:50] = 99.0  # wreck the first half
    mask = np.zeros(100, dtype=bool)
    mask[50:] = True  # score only the intact half
    assert fvu(t, recon, mask=mask) == 0.0


def test_fvu_empty_is_nan():
    t = _data()
    assert np.isnan(fvu(t, t.copy(), mask=np.zeros(100, dtype=bool)))


def test_fvu_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        fvu(_data(), _data()[:, :4])


def test_fvu_by_kind_splits_text_image():
    t = _data()
    recon = t.copy()
    kinds = np.array(["text"] * 50 + ["image"] * 50)
    recon[kinds == "image"] = 99.0  # only image positions are wrecked
    out = fvu_by_kind(t, recon, kinds)
    assert out["text"] == 0.0
    assert out["image"] > 1.0
    assert "all" in out


def test_l0_counts_active_features():
    feats = np.zeros((10, 100))
    feats[:, :5] = 1.0  # exactly 5 active per position
    assert l0(feats) == 5.0


def test_l0_handles_layer_axes():
    feats = np.zeros((3, 10, 100))  # (layers, pos, features)
    feats[..., :7] = 2.0
    assert l0(feats) == 7.0


def test_summarize_is_flat_scorecard():
    t = _data()
    kinds = np.array(["text"] * 60 + ["image"] * 40)
    feats = np.zeros((100, 50))
    feats[:, :3] = 1.0
    out = summarize(t, t.copy(), kinds, feature_acts=feats)
    assert out["fvu_text"] == 0.0 and out["fvu_image"] == 0.0
    assert out["fvu_all"] == 0.0
    assert out["l0"] == 3.0
