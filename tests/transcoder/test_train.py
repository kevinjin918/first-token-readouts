"""Tests for the CLT training loop.

``load_stacked`` (NumPy) runs in CI. ``train_clt`` needs torch (the CLT), so it's guarded by
``importorskip``. We build a tiny on-disk cache where each layer's MLP output is a *learnable*
function of its input, then confirm training drives reconstruction error down (FVU < 1).
"""

from __future__ import annotations

import numpy as np
import pytest
from tracecxr.transcoder.cache import RecordActivations, cache_corpus
from tracecxr.transcoder.clt import CLTConfig
from tracecxr.transcoder.train import load_stacked, train_clt

LAYERS = [0, 1, 2]
D = 8


def _learnable_cache(tmp_path, n_text=24, n_image=12):
    """Cache where out = in @ W_layer (+ small noise) — a target the CLT can fit."""
    rng = np.random.default_rng(0)
    w = {L: rng.standard_normal((D, D)) for L in LAYERS}

    def capture(record) -> RecordActivations:
        has_image = getattr(record, "image", None) is not None
        seq = 7 if has_image else 5
        mlp_in = {L: rng.standard_normal((seq, D)) for L in LAYERS}
        mlp_out = {L: mlp_in[L] @ w[L] + 0.01 * rng.standard_normal((seq, D)) for L in LAYERS}
        kinds = np.array(["image" if has_image else "text"] * seq)
        return RecordActivations(str(getattr(record, "id", "r")), mlp_in, mlp_out, kinds)

    class _Rec:
        def __init__(self, rid, image=None):
            self.id, self.image = rid, image

    records = [_Rec(f"t{i}") for i in range(n_text)] + [
        _Rec(f"i{i}", image=object()) for i in range(n_image)
    ]
    cache_corpus(records, LAYERS, tmp_path, capture, shard_size=8)
    return tmp_path


def test_load_stacked_shapes(tmp_path):
    _learnable_cache(tmp_path)
    x, y, kinds, layers = load_stacked(tmp_path)
    # 24 text * 5 + 12 image * 7 = 204 tokens; 3 layers; d_model 8
    assert x.shape == (204, 3, D)
    assert y.shape == (204, 3, D)
    assert kinds.shape == (204,)
    assert layers == LAYERS
    assert set(kinds.tolist()) == {"text", "image"}


def test_load_stacked_respects_max_tokens(tmp_path):
    _learnable_cache(tmp_path)
    x, _, _, _ = load_stacked(tmp_path, max_tokens=50)
    assert x.shape[0] == 50


def test_load_stacked_max_tokens_stays_balanced(tmp_path):
    """A max_tokens cap must sample across kinds, not just the (kind-ordered) early shards."""
    _learnable_cache(tmp_path)  # caches text records first, then image records
    _, _, kinds, _ = load_stacked(tmp_path, max_tokens=50, seed=0)
    # both kinds present in the capped subset — a manifest-order truncation would be text-only
    assert set(kinds.tolist()) == {"text", "image"}


def test_train_clt_reduces_fvu(tmp_path):
    pytest.importorskip("torch")
    _learnable_cache(tmp_path)
    cfg = CLTConfig(n_features=64, k=8, span=1, learning_rate=1e-2, device="cpu")
    out = train_clt(tmp_path, cfg, steps=400, batch_size=64, eval_every=200, seed=0)

    assert "clt" in out and out["layers"] == LAYERS
    assert out["clt"].config.n_layers == 3  # set from the cache
    sc = out["scorecard"]
    assert {"fvu_all", "fvu_text", "fvu_image"} <= set(sc)
    # the target is learnable -> FVU should drop well below the mean-baseline of 1.0
    assert sc["fvu_all"] < 0.5
    # and it should improve over training
    assert out["history"][-1]["recon_mse"] < out["history"][0]["recon_mse"]


def test_checkpoint_resume(tmp_path):
    """A second call with the same checkpoint dir resumes instead of restarting (spot safety)."""
    pytest.importorskip("torch")
    cache = tmp_path / "cache"
    _learnable_cache(cache)
    ckpt = tmp_path / "ckpt"
    kw = dict(n_features=64, k=8, span=1, learning_rate=1e-2, device="cpu")

    train_clt(cache, CLTConfig(**kw), steps=200, batch_size=64, eval_every=100,
              checkpoint_dir=ckpt, checkpoint_every=100, seed=0)
    assert (ckpt / "clt_ckpt.pt").exists()

    # ask for 400 total -> must resume from 200 and run only 201..400
    out2 = train_clt(cache, CLTConfig(**kw), steps=400, batch_size=64, eval_every=100,
                     checkpoint_dir=ckpt, checkpoint_every=100, resume=True, seed=0)
    steps_run = [h["step"] for h in out2["history"]]
    assert min(steps_run) > 200 and max(steps_run) == 400
