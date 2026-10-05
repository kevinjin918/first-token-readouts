"""Tests for the Stage 1 cross-layer transcoder (manifesto §Stage 1).

Config/interface tests run in CI with no torch. The compute behavior (encode/decode/
reconstruct/train_step) is exercised at tiny scale and guarded by ``importorskip("torch")``,
since torch is the optional ``[models]`` extra.
"""

from __future__ import annotations

import pytest
from tracecxr.core import config
from tracecxr.transcoder import CLTConfig, CrossLayerTranscoder

# --- Config / interface (no torch) ----------------------------------------------------


def test_module_imports_cleanly() -> None:
    import tracecxr.transcoder as pkg

    assert set(pkg.__all__) == {"CLTConfig", "CrossLayerTranscoder"}


def test_config_defaults() -> None:
    cfg = CLTConfig()
    assert cfg.jumprelu is True
    assert cfg.lambda_ > 0
    assert cfg.c > 0
    assert cfg.learning_rate > 0
    assert cfg.batch_size > 0
    assert cfg.n_features > 0
    assert cfg.n_layers > 0
    assert 10 <= cfg.l0_target <= 300
    assert 0.0 <= cfg.recon_error_target <= 0.2
    assert cfg.k > 0


def test_default_warm_start_references_released_transcoder() -> None:
    cfg = CLTConfig()
    assert cfg.warm_start_from == config.TRANSCODERS["gemma3-4b-it-clt"].name


def test_config_accepts_overrides() -> None:
    cfg = CLTConfig(
        jumprelu=False, lambda_=5e-4, c=2.0, learning_rate=3e-4, batch_size=512,
        n_features=4096, n_layers=12, l0_target=88, recon_error_target=0.15,
        warm_start_from=None, span=0, k=16, extra={"note": "windowed"},
    )
    assert cfg.jumprelu is False
    assert cfg.n_features == 4096
    assert cfg.n_layers == 12
    assert cfg.span == 0
    assert cfg.k == 16
    assert cfg.warm_start_from is None


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("n_features", 0),
        ("n_layers", 0),
        ("l0_target", 0),
        ("recon_error_target", 1.5),
        ("recon_error_target", -0.1),
        ("span", -1),
        ("k", 0),
    ],
)
def test_config_rejects_invalid_values(field_name: str, value: float) -> None:
    with pytest.raises(ValueError):
        CLTConfig(**{field_name: value})


def test_k_must_not_exceed_n_features() -> None:
    with pytest.raises(ValueError, match="k must be"):
        CLTConfig(n_features=16, k=32)


def test_transcoder_defaults_to_fresh_config() -> None:
    clt = CrossLayerTranscoder()
    assert isinstance(clt.config, CLTConfig)
    assert clt.config == CLTConfig()


def test_transcoder_holds_supplied_config_and_repr() -> None:
    cfg = CLTConfig(n_features=4096, n_layers=12, span=2)
    clt = CrossLayerTranscoder(cfg)
    assert clt.config is cfg
    assert "n_features=4096" in repr(clt)
    assert "n_layers=12" in repr(clt)
    assert "span=2" in repr(clt)


def test_load_warm_start_unsupported() -> None:
    """Warm-start is intentionally unimplemented (A1 convention mismatch)."""
    clt = CrossLayerTranscoder()
    with pytest.raises(NotImplementedError, match="warm-start"):
        clt.load_warm_start()


# --- Compute behavior (needs torch) ---------------------------------------------------


def _tiny() -> CrossLayerTranscoder:
    cfg = CLTConfig(n_layers=3, n_features=16, k=4, span=1, learning_rate=1e-2, device="cpu")
    return CrossLayerTranscoder(cfg)


def test_encode_is_topk_sparse() -> None:
    torch = pytest.importorskip("torch")
    clt = _tiny()
    x = torch.randn(5, 3, 8)  # (batch, n_layers, d_model)
    feats = clt.encode(x)
    assert feats.shape == (5, 3, 16)
    # exactly k=4 active features per (position, layer)
    assert int((feats > 0).sum(-1).max()) <= 4
    assert int((feats > 0).sum(-1).min()) >= 1


def test_reconstruct_shape_matches_mlp_out() -> None:
    torch = pytest.importorskip("torch")
    clt = _tiny()
    x = torch.randn(5, 3, 8)
    recon = clt.reconstruct(x)
    assert recon.shape == (5, 3, 8)


def test_span_zero_is_per_layer() -> None:
    """With span=0 a source layer's features write only to their own layer."""
    torch = pytest.importorskip("torch")
    cfg = CLTConfig(n_layers=3, n_features=16, k=4, span=0, device="cpu")
    clt = CrossLayerTranscoder(cfg)
    clt.encode(torch.randn(2, 3, 8))  # triggers the lazy build
    # decoder is split per write-offset; span 0 -> exactly one offset tensor (writes own layer only)
    assert "W_dec_0" in clt._params and "W_dec_1" not in clt._params
    assert clt._params["W_dec_0"].shape == (3, 16, 8)  # (n_layers, n_features, d_model)


def test_decoder_split_one_tensor_per_offset() -> None:
    """span S -> S+1 per-offset decoder tensors; span=None -> full cross-layer (n_layers)."""
    torch = pytest.importorskip("torch")
    clt = CrossLayerTranscoder(CLTConfig(n_layers=5, n_features=16, k=4, span=2, device="cpu"))
    clt.encode(torch.randn(2, 5, 8))
    dec_keys = sorted(k for k in clt._params if k.startswith("W_dec_"))
    assert dec_keys == ["W_dec_0", "W_dec_1", "W_dec_2"]  # span+1 tensors
    assert all(clt._params[k].shape == (5, 16, 8) for k in dec_keys)  # none has a span axis

    full = CrossLayerTranscoder(CLTConfig(n_layers=5, n_features=16, k=4, span=None, device="cpu"))
    full.encode(torch.randn(2, 5, 8))
    # full cross-layer: a feature can write to every downstream layer -> n_layers offsets (0..4)
    full_keys = sorted(k for k in full._params if k.startswith("W_dec_"))
    assert full_keys == [f"W_dec_{o}" for o in range(5)]


def test_train_step_reduces_loss() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    clt = _tiny()
    # a fixed, learnable target: y = x @ W (per layer) -> the CLT should overfit it.
    x = torch.randn(64, 3, 8)
    w = torch.randn(8, 8)
    y = x @ w
    first = clt.train_step((x, y))["recon_mse"]
    for _ in range(300):
        last = clt.train_step((x, y))
    assert last["recon_mse"] < first * 0.5  # loss at least halves
    assert last["l0"] <= 4.0  # TopK sparsity holds


def test_amp_is_noop_on_cpu_and_still_trains() -> None:
    """amp=True must fall back cleanly on CPU (no CUDA autocast) and still optimize."""
    torch = pytest.importorskip("torch")
    cfg = CLTConfig(n_layers=3, n_features=16, k=4, span=1, learning_rate=1e-2,
                    device="cpu", amp=True)
    clt = CrossLayerTranscoder(cfg)
    x = torch.randn(32, 3, 8)
    y = x @ torch.randn(8, 8)
    first = clt.train_step((x, y))["recon_mse"]
    for _ in range(200):
        last = clt.train_step((x, y))
    assert last["recon_mse"] < first


def test_checkpoint_roundtrip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    clt = _tiny()
    x, y = torch.randn(8, 3, 8), torch.randn(8, 3, 8)
    for _ in range(5):
        clt.train_step((x, y))
    path = str(tmp_path / "c.pt")
    clt.save_checkpoint(path, step=5)

    restored = _tiny()
    assert restored.load_checkpoint(path) == 5
    for key in clt._params:
        assert torch.allclose(clt._params[key], restored._params[key])


def test_default_optimizer_is_torch_adam() -> None:
    torch = pytest.importorskip("torch")
    clt = _tiny()
    clt.encode(torch.randn(2, 3, 8))  # triggers the lazy build (params + optimizer)
    assert isinstance(clt._opt, torch.optim.Adam)


def test_adam_8bit_falls_back_without_bitsandbytes() -> None:
    torch = pytest.importorskip("torch")
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("bitsandbytes") is not None:
        pytest.skip("bitsandbytes installed; the 8-bit path is taken, fallback not exercised")
    cfg = CLTConfig(n_layers=2, n_features=8, k=2, span=0, device="cpu", adam_8bit=True)
    clt = CrossLayerTranscoder(cfg)
    with pytest.warns(UserWarning, match="bitsandbytes"):
        clt.encode(torch.randn(1, 2, 4))  # build -> _build_optimizer -> fallback warning
    assert isinstance(clt._opt, torch.optim.Adam)
