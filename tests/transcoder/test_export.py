"""Tests for the weights-only safetensors export of a CLT checkpoint."""

from __future__ import annotations

import pytest
from tracecxr.transcoder import CLTConfig, CrossLayerTranscoder


def _trained(span: int | None) -> CrossLayerTranscoder:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    cfg = CLTConfig(n_layers=4, n_features=16, k=4, span=span, learning_rate=1e-2, device="cpu")
    clt = CrossLayerTranscoder(cfg)
    x, y = torch.randn(8, 4, 8), torch.randn(8, 4, 8)
    for _ in range(3):
        clt.train_step((x, y))
    return clt


@pytest.mark.parametrize("span", [None, 1, 0])
def test_export_reproduces_the_model(tmp_path, span) -> None:
    torch = pytest.importorskip("torch")
    from tracecxr.transcoder.export import export_safetensors, verify_export

    clt = _trained(span)
    ckpt, out = tmp_path / "c.pt", tmp_path / "c.safetensors"
    clt.save_checkpoint(str(ckpt), step=3)
    summary = export_safetensors(ckpt, out)
    assert verify_export(ckpt, out) == 3 + (clt._span + 1)

    # only the decoder rows decode reads are kept
    n, f, d = 4, 16, 8
    kept = n * d * f + n * f + n * d + sum((n - o) * f * d for o in range(clt._span + 1))
    assert summary["n_params"] == kept

    restored = CrossLayerTranscoder(clt.config)
    assert restored.load_checkpoint(str(out), load_optimizer=False) == 3
    x = torch.randn(5, 4, 8)
    assert torch.equal(clt.reconstruct(x), restored.reconstruct(x))


def test_export_cannot_resume_training(tmp_path) -> None:
    from tracecxr.transcoder.export import export_safetensors

    clt = _trained(1)
    ckpt, out = tmp_path / "c.pt", tmp_path / "c.safetensors"
    clt.save_checkpoint(str(ckpt), step=3)
    export_safetensors(ckpt, out)
    with pytest.raises(ValueError, match="weights only"):
        CrossLayerTranscoder(clt.config).load_checkpoint(str(out))


def test_verify_catches_a_changed_tensor(tmp_path) -> None:
    from tracecxr.transcoder.export import export_safetensors, verify_export

    clt = _trained(1)
    ckpt, out = tmp_path / "c.pt", tmp_path / "c.safetensors"
    clt.save_checkpoint(str(ckpt), step=3)
    export_safetensors(ckpt, out)
    clt._params["W_dec_1"].data[0, 0, 0] += 1.0
    clt.save_checkpoint(str(ckpt), step=3)
    with pytest.raises(AssertionError, match="W_dec_1"):
        verify_export(ckpt, out)
