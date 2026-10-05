"""Tests for the activation-cache pipeline orchestration.

The real MedGemma capture path needs a GPU (covered by requires_gpu elsewhere); here we inject
a mock ``capture_fn`` so the sharding, manifest, and round-trip read are fully testable in CI.
"""

from __future__ import annotations

import numpy as np
from tracecxr.transcoder.cache import (
    RecordActivations,
    cache_corpus,
    decode_act,
    iter_shards,
    load_layer,
    load_manifest,
)

LAYERS = [0, 1]


def _mock_capture_fn(seq_text: int = 5, seq_image: int = 7):
    rng = np.random.default_rng(0)

    def capture(record) -> RecordActivations:
        has_image = getattr(record, "image", None) is not None
        n = seq_image if has_image else seq_text
        kinds = np.array(["image" if has_image else "text"] * n)
        return RecordActivations(
            record_id=str(getattr(record, "id", "r")),
            mlp_in={L: rng.standard_normal((n, 4)) for L in LAYERS},
            mlp_out={L: rng.standard_normal((n, 4)) for L in LAYERS},
            kinds=kinds,
        )

    return capture


class _Rec:
    def __init__(self, rid, image=None):
        self.id = rid
        self.image = image


def test_cache_corpus_writes_shards_and_manifest(tmp_path):
    records = [_Rec(f"t{i}") for i in range(5)] + [_Rec(f"i{i}", image=object()) for i in range(3)]
    manifest = cache_corpus(records, LAYERS, tmp_path, _mock_capture_fn(), shard_size=4)

    assert manifest["n_records"] == 8
    assert manifest["layers"] == LAYERS
    assert len(manifest["shards"]) == 2  # 8 records / shard_size 4
    # 5 text records * 5 tokens, 3 image records * 7 tokens
    assert manifest["token_counts"]["text"] == 25
    assert manifest["token_counts"]["image"] == 21
    assert load_manifest(tmp_path)["n_records"] == 8


def test_cache_shard_arrays_have_expected_shapes(tmp_path):
    records = [_Rec(f"t{i}") for i in range(4)]
    cache_corpus(records, LAYERS, tmp_path, _mock_capture_fn(), shard_size=4)
    (shard,) = list(iter_shards(tmp_path))
    assert shard["in_0"].shape == (20, 4)  # 4 records * 5 text tokens
    assert shard["out_1"].shape == (20, 4)
    assert shard["kinds"].shape == (20,)


def test_load_layer_round_trips_across_shards(tmp_path):
    records = [_Rec(f"t{i}") for i in range(6)]  # -> 2 shards at shard_size 4
    cache_corpus(records, LAYERS, tmp_path, _mock_capture_fn(), shard_size=4)
    mlp_in, mlp_out, kinds = load_layer(tmp_path, layer=0)
    assert mlp_in.shape == (30, 4)  # 6 records * 5 tokens
    assert mlp_out.shape == (30, 4)
    assert set(kinds.tolist()) == {"text"}


def test_partial_final_shard(tmp_path):
    records = [_Rec(f"t{i}") for i in range(5)]  # shard_size 4 -> shards of 4 and 1
    manifest = cache_corpus(records, LAYERS, tmp_path, _mock_capture_fn(), shard_size=4)
    assert len(manifest["shards"]) == 2
    assert manifest["token_counts"]["text"] == 25


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit pattern as int16 (round-to-nearest-even, matching torch's .bfloat16())."""
    u = x.astype(np.float32).view(np.uint32)
    # round-to-nearest-even on the truncated low 16 bits
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16).view(np.int16)


def test_decode_act_bf16_roundtrip_and_fp32_passthrough():
    x = np.array([[0.0, 1.0, -2.5, 1000.0, -65504.0, 1e-3]], dtype=np.float32)
    decoded = decode_act(_bf16_bits(x))
    assert decoded.dtype == np.float32 and decoded.shape == x.shape
    # bf16 has 8 mantissa bits -> ~1/256 relative error; values round-trip within that
    np.testing.assert_allclose(decoded, x, rtol=1 / 256, atol=1e-4)
    # fp32 arrays pass through unchanged
    np.testing.assert_array_equal(decode_act(x), x)


def test_bf16_cache_round_trips_through_load_layer(tmp_path):
    """A cache written as bf16-bits (int16) must read back as fp32 via the decode-on-load path."""
    rng = np.random.default_rng(0)
    vals = {L: rng.standard_normal((5, 4)).astype(np.float32) for L in LAYERS}

    def capture(record) -> RecordActivations:
        return RecordActivations(
            record_id=str(record.id),
            mlp_in={L: _bf16_bits(vals[L]) for L in LAYERS},  # stored as int16 bf16-bits
            mlp_out={L: _bf16_bits(vals[L]) for L in LAYERS},
            kinds=np.array(["text"] * 5),
        )

    cache_corpus([_Rec("t0")], LAYERS, tmp_path, capture, shard_size=4)
    mlp_in, mlp_out, kinds = load_layer(tmp_path, layer=0)
    assert mlp_in.dtype == np.float32  # decoded, not raw int16
    np.testing.assert_allclose(mlp_in, vals[0], rtol=1 / 256, atol=1e-4)
    assert set(kinds.tolist()) == {"text"}


def test_cache_resume_skips_existing_and_continues(tmp_path):
    records = [_Rec(f"t{i}") for i in range(16)]
    cap = _mock_capture_fn()
    calls = {"n": 0}

    def counting(rec):
        calls["n"] += 1
        return cap(rec)

    cache_corpus(records[:8], LAYERS, tmp_path, counting, shard_size=4)  # 2 shards
    assert calls["n"] == 8
    calls["n"] = 0
    man = cache_corpus(records, LAYERS, tmp_path, counting, shard_size=4, resume=True)
    assert calls["n"] == 8  # only the 8 new records were captured; the first 8 were skipped
    assert len(man["shards"]) == 4
    assert man["n_records"] == 16
    assert man["token_counts"]["text"] == 80  # all 16 text records * 5 tokens (existing recounted)


def test_cache_resume_false_reprocesses(tmp_path):
    records = [_Rec(f"t{i}") for i in range(8)]
    cache_corpus(records, LAYERS, tmp_path, _mock_capture_fn(), shard_size=4)
    cap = _mock_capture_fn()
    calls = {"n": 0}

    def counting(rec):
        calls["n"] += 1
        return cap(rec)

    man = cache_corpus(records, LAYERS, tmp_path, counting, shard_size=4, resume=False)
    assert calls["n"] == 8  # resume off -> reprocess everything from scratch
    assert len(man["shards"]) == 2
