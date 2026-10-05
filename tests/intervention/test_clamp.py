"""CI-testable surface of the CLT clamp engine (manifesto §Stage 4).

The :class:`ClampedMedGemma` runtime needs MedGemma + a CLT checkpoint on a GPU and runs
only on the H100. Here we cover the pure feature-edit core and the spec validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from tracecxr.intervention import ClampedMedGemma, FeatureEdit, apply_feature_edits

LAYERS = [8, 9, 10, 11]  # model layers in CLT index order


def _feats() -> np.ndarray:
    # (seq=3, n_layers=4, n_features=5), all ones so edits are easy to read off.
    return np.ones((3, 4, 5), dtype=np.float32)


def test_featureedit_validates_op_and_positions() -> None:
    with pytest.raises(ValueError, match="op must be one of"):
        FeatureEdit(layer=8, feature=0, op="nope")
    with pytest.raises(ValueError, match="positions must be one of"):
        FeatureEdit(layer=8, feature=0, positions="middle")


def test_featureedit_defaults_to_negative_steering() -> None:
    e = FeatureEdit(layer=9, feature=2)
    assert e.op == "scale" and e.amount == -1.0 and e.positions == "all"


def test_ablate_zeros_the_feature_everywhere() -> None:
    feats = _feats()
    out = apply_feature_edits(feats, [FeatureEdit(8, 2, op="ablate")], LAYERS)
    assert np.all(out[:, 0, 2] == 0.0)  # layer 8 -> idx 0, feature 2 zeroed at all positions
    # nothing else touched
    assert out.sum() == feats.sum() - feats[:, 0, 2].sum()


def test_scale_negative_steers() -> None:
    out = apply_feature_edits(_feats(), [FeatureEdit(10, 1, op="scale", amount=-1.0)], LAYERS)
    assert np.all(out[:, 2, 1] == -1.0)  # layer 10 -> idx 2


def test_set_forces_value() -> None:
    out = apply_feature_edits(_feats(), [FeatureEdit(11, 4, op="set", amount=3.5)], LAYERS)
    assert np.all(out[:, 3, 4] == 3.5)


def test_does_not_mutate_input() -> None:
    feats = _feats()
    apply_feature_edits(feats, [FeatureEdit(8, 0, op="ablate")], LAYERS)
    assert np.all(feats == 1.0)


def test_unknown_layer_raises() -> None:
    with pytest.raises(KeyError, match="not in the CLT's layers"):
        apply_feature_edits(_feats(), [FeatureEdit(99, 0, op="ablate")], LAYERS)


def test_image_positions_need_mask() -> None:
    with pytest.raises(ValueError, match="needs an img_mask"):
        apply_feature_edits(_feats(), [FeatureEdit(8, 0, positions="image")], LAYERS)


def test_image_and_text_position_restriction() -> None:
    feats = _feats()
    mask = np.array([True, False, True])  # positions 0 and 2 are image patches
    img = apply_feature_edits(feats, [FeatureEdit(8, 0, op="ablate", positions="image")],
                              LAYERS, img_mask=mask)
    assert list(img[:, 0, 0]) == [0.0, 1.0, 0.0]  # only image rows zeroed
    txt = apply_feature_edits(feats, [FeatureEdit(8, 0, op="set", amount=9.0, positions="text")],
                              LAYERS, img_mask=mask)
    assert list(txt[:, 0, 0]) == [1.0, 9.0, 1.0]  # only the text row set


def test_multiple_edits_apply_in_order() -> None:
    out = apply_feature_edits(
        _feats(),
        [FeatureEdit(8, 0, op="set", amount=2.0), FeatureEdit(8, 0, op="scale", amount=3.0)],
        LAYERS,
    )
    assert np.all(out[:, 0, 0] == 6.0)  # set to 2 then scaled by 3


def test_runtime_construction_is_lazy() -> None:
    runtime = ClampedMedGemma("ckpt.pt", LAYERS)
    assert runtime._state == {}


@pytest.mark.requires_gpu
def test_clamp_on_real_medgemma() -> None:  # pragma: no cover - GPU only
    pytest.skip("needs MedGemma weights + a CLT checkpoint on a GPU; run on the H100")


def test_prefill_hook_substitutes_prefill_and_passes_decode_through() -> None:
    torch = pytest.importorskip("torch")
    from tracecxr.intervention.clamp import prefill_substitute_hook

    plen, d = 5, 3
    recon = torch.arange(plen * d, dtype=torch.float32).reshape(plen, d)
    hook = prefill_substitute_hook(recon, plen)
    # prefill, batch 4 (num_return_sequences): replaced, broadcast over the batch, dtype kept
    out = hook(None, None, torch.zeros(4, plen, d, dtype=torch.bfloat16))
    assert out.shape == (4, plen, d) and out.dtype == torch.bfloat16
    assert torch.equal(out[3].float(), recon)
    # decode step: returns None so the live MLP output is kept
    assert hook(None, None, torch.zeros(4, 1, d)) is None


def test_prefill_add_hook_adds_on_prefill_and_passes_decode_through() -> None:
    torch = pytest.importorskip("torch")
    from tracecxr.intervention.clamp import _prefill_hook, prefill_add_hook

    plen, d = 5, 3
    delta = torch.zeros(plen, d)
    delta[1] = 1.0  # one edited (image) position
    live = torch.randn(4, plen, d)
    out = prefill_add_hook(delta, plen)(None, None, live)
    # the live output is kept everywhere and shifted only where the edit is nonzero
    assert torch.equal(out[:, [0, 2, 3, 4]], live[:, [0, 2, 3, 4]])
    assert torch.allclose(out[:, 1], live[:, 1] + 1.0)
    # no edit: the plain model
    assert torch.equal(prefill_add_hook(torch.zeros(plen, d), plen)(None, None, live), live)
    assert prefill_add_hook(delta, plen)(None, None, torch.zeros(4, 1, d)) is None
    with pytest.raises(ValueError):
        _prefill_hook("freeze")
