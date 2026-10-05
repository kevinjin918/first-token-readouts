"""Tests for the streaming activation loader (docs/clt_plan.md).

``split_shards`` and ``stream_balanced_batches`` are pure NumPy and run in CI. The streaming
trainer ``train_clt_streaming`` needs torch (the CLT), so it's guarded by ``importorskip``.
Mock shards encode the token kind in ``out[:, 0]`` (text=0, image=1) so balance is checkable
from the yielded batches.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from tracecxr.transcoder.stream import split_shards, stream_balanced_batches

LAYERS = [0, 1, 2]
D = 6


def _write_cache(tmp_path, n_shards=6, seq=10):
    """Write ``n_shards`` mock shards (kind encoded in out[:, 0]) + a manifest."""
    rng = np.random.default_rng(0)
    names = []
    for s in range(n_shards):
        kinds = np.array(["text"] * (seq // 2) + ["image"] * (seq - seq // 2))
        payload = {}
        for layer in LAYERS:
            payload[f"in_{layer}"] = rng.standard_normal((seq, D)).astype(np.float32)
            out = rng.standard_normal((seq, D)).astype(np.float32)
            out[:, 0] = (kinds == "image").astype(np.float32)  # encode kind for balance checks
            payload[f"out_{layer}"] = out
        payload["kinds"] = kinds
        name = f"shard_{s:05d}.npz"
        np.savez(tmp_path / name, **payload)
        names.append(name)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"layers": LAYERS, "shards": names, "n_records": n_shards,
                    "shard_size": 1, "token_counts": {}})
    )
    return names


def test_split_shards_disjoint_and_covers(tmp_path):
    names = _write_cache(tmp_path)
    train, test = split_shards({"shards": names}, test_frac=0.25, seed=0)
    assert set(train) | set(test) == set(names)
    assert set(train) & set(test) == set()  # no leakage
    assert train and test


def test_split_shards_requires_two(tmp_path):
    with pytest.raises(ValueError, match=">= 2 shards"):
        split_shards({"shards": ["only.npz"]})


def test_stream_shapes_and_count(tmp_path):
    names = _write_cache(tmp_path)
    batches = list(stream_balanced_batches(tmp_path, names, LAYERS, batch_size=8, steps=5,
                                           buffer_shards=2, steps_per_buffer=2))
    assert len(batches) == 5
    for x, y in batches:
        assert x.shape == (8, len(LAYERS), D)  # 2 kinds * (8 // 2)
        assert y.shape == (8, len(LAYERS), D)


def test_stream_is_balanced(tmp_path):
    names = _write_cache(tmp_path)
    for _x, y in stream_balanced_batches(tmp_path, names, LAYERS, batch_size=8, steps=4,
                                         buffer_shards=2, steps_per_buffer=2):
        # out[:, 0, 0] encodes kind (text=0, image=1) — a balanced batch holds both
        assert set(np.unique(y[:, 0, 0]).tolist()) == {0.0, 1.0}


def test_stream_refills_across_buffers(tmp_path):
    # buffer smaller than the corpus + more steps than one buffer -> must evict & reload
    names = _write_cache(tmp_path, n_shards=6)
    batches = list(stream_balanced_batches(tmp_path, names, LAYERS, batch_size=4, steps=12,
                                           buffer_shards=2, steps_per_buffer=3))
    assert len(batches) == 12


def test_stream_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no shards"):
        next(stream_balanced_batches(tmp_path, [], LAYERS, batch_size=4, steps=1))


def test_train_clt_streaming_runs(tmp_path):
    pytest.importorskip("torch")
    import random

    from tracecxr.transcoder.cache import RecordActivations, cache_corpus
    from tracecxr.transcoder.clt import CLTConfig
    from tracecxr.transcoder.train import train_clt_streaming

    rng = np.random.default_rng(0)
    w = {L: rng.standard_normal((D, D)) for L in LAYERS}

    def capture(record):  # out = in @ W_layer — a target the CLT can fit
        has_image = getattr(record, "image", None) is not None
        seq = 7 if has_image else 5
        mlp_in = {L: rng.standard_normal((seq, D)) for L in LAYERS}
        mlp_out = {L: mlp_in[L] @ w[L] for L in LAYERS}
        kinds = np.array(["image" if has_image else "text"] * seq)
        return RecordActivations(str(getattr(record, "id", "r")), mlp_in, mlp_out, kinds)

    class _Rec:
        def __init__(self, rid, image=None):
            self.id, self.image = rid, image

    records = [_Rec(f"t{i}") for i in range(24)]
    records += [_Rec(f"i{i}", image=object()) for i in range(12)]
    random.Random(0).shuffle(records)  # mix kinds across shards, as production caching does
    cache_corpus(records, LAYERS, tmp_path, capture, shard_size=8)  # ~5 shards

    cfg = CLTConfig(n_features=64, k=8, span=1, learning_rate=1e-2, device="cpu")
    out = train_clt_streaming(tmp_path, cfg, steps=400, batch_size=64, eval_every=200,
                              buffer_shards=2, steps_per_buffer=100)
    assert "fvu_text" in out["scorecard"] and "fvu_image" in out["scorecard"]
    assert out["scorecard"]["fvu_text"] < 1.0  # learns the linear target
    assert len(out["layers"]) == len(LAYERS)


def test_train_clt_streaming_caps_held_out(tmp_path, monkeypatch):
    """max_test_tokens must reach the held-out load — else a 10% split of a huge cache OOMs."""
    pytest.importorskip("torch")
    import random

    from tracecxr.transcoder import train as train_mod
    from tracecxr.transcoder.cache import RecordActivations, cache_corpus
    from tracecxr.transcoder.clt import CLTConfig
    from tracecxr.transcoder.train import train_clt_streaming

    rng = np.random.default_rng(0)
    w = {L: rng.standard_normal((D, D)) for L in LAYERS}

    def capture(record):
        has_image = getattr(record, "image", None) is not None
        seq = 7 if has_image else 5
        mlp_in = {L: rng.standard_normal((seq, D)) for L in LAYERS}
        mlp_out = {L: mlp_in[L] @ w[L] for L in LAYERS}
        kinds = np.array(["image" if has_image else "text"] * seq)
        return RecordActivations(str(getattr(record, "id", "r")), mlp_in, mlp_out, kinds)

    class _Rec:
        def __init__(self, rid, image=None):
            self.id, self.image = rid, image

    records = [_Rec(f"t{i}") for i in range(24)]
    records += [_Rec(f"i{i}", image=object()) for i in range(12)]
    random.Random(0).shuffle(records)
    cache_corpus(records, LAYERS, tmp_path, capture, shard_size=8)

    # spy on load_stacked to confirm the cap propagates to the held-out read.
    seen: dict = {}
    real = train_mod.load_stacked

    def spy(cache_dir, **kw):
        seen.update(kw)
        return real(cache_dir, **kw)

    monkeypatch.setattr(train_mod, "load_stacked", spy)

    cfg = CLTConfig(n_features=64, k=8, span=1, learning_rate=1e-2, device="cpu")
    out = train_clt_streaming(tmp_path, cfg, steps=50, batch_size=32, eval_every=50,
                              buffer_shards=2, steps_per_buffer=50, max_test_tokens=12)
    assert seen.get("max_tokens") == 12  # the cap reached the held-out load (the OOM guard)
    assert any(k.startswith("fvu_") for k in out["scorecard"])  # still scores on the capped set


def test_progress_fn_fires_incrementally(tmp_path):
    """progress_fn is called DURING training -- full eval records + light heartbeats between."""
    pytest.importorskip("torch")
    import random

    from tracecxr.transcoder.cache import RecordActivations, cache_corpus
    from tracecxr.transcoder.clt import CLTConfig
    from tracecxr.transcoder.train import train_clt_streaming

    rng = np.random.default_rng(0)
    w = {L: rng.standard_normal((D, D)) for L in LAYERS}

    def capture(record):
        has_image = getattr(record, "image", None) is not None
        seq = 7 if has_image else 5
        mlp_in = {L: rng.standard_normal((seq, D)) for L in LAYERS}
        mlp_out = {L: mlp_in[L] @ w[L] for L in LAYERS}
        kinds = np.array(["image" if has_image else "text"] * seq)
        return RecordActivations(str(getattr(record, "id", "r")), mlp_in, mlp_out, kinds)

    class _Rec:
        def __init__(self, rid, image=None):
            self.id, self.image = rid, image

    records = [_Rec(f"t{i}") for i in range(24)]
    records += [_Rec(f"i{i}", image=object()) for i in range(12)]
    random.Random(0).shuffle(records)
    cache_corpus(records, LAYERS, tmp_path, capture, shard_size=8)

    seen: list = []
    cfg = CLTConfig(n_features=64, k=8, span=1, learning_rate=1e-2, device="cpu")
    train_clt_streaming(tmp_path, cfg, steps=20, batch_size=32, eval_every=10,
                        buffer_shards=2, steps_per_buffer=20,
                        progress_fn=lambda r: seen.append(r), heartbeat_every=5)
    evals = [r for r in seen if any(k.startswith("fvu_") for k in r)]
    hbs = [r for r in seen if not any(k.startswith("fvu_") for k in r)]
    assert any(r["step"] == 10 for r in evals)  # full eval record mid-run (step 10)
    assert any(r["step"] == 5 for r in hbs)      # heartbeat between evals (step 5, no scorecard)
