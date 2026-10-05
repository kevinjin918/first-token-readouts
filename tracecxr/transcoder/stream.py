"""Streaming activation loader for large-scale CLT training (docs/clt_plan.md).

B1 stacked the whole cache in memory (fine for ~250k tokens). The release-scale CLT trains on
tens of millions of tokens that won't fit, so this streams sharded ``.npz`` activations from disk
through a **bounded shuffle buffer**, yielding text/image-balanced batches (the A1/A3 lesson)
without ever holding the full corpus in memory.

Pure NumPy and fully testable. The streaming trainer
(:func:`tracecxr.transcoder.train.train_clt_streaming`) moves each yielded batch to the GPU per
step — at release scale the model's forward/backward dominates, so per-batch host->device copy is
cheap (unlike the tiny-dictionary L4 case, where preloading mattered).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tracecxr.transcoder.cache import decode_act


def split_shards(
    manifest: dict[str, Any], *, test_frac: float = 0.1, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Partition the manifest's shards into ``(train, test)`` filename lists.

    Splitting by *shard* (not by token) keeps the held-out set fully disjoint from training — no
    token from a test shard is ever streamed for training. Reserves at least one test shard.

    Raises:
        ValueError: if there are fewer than 2 shards (nothing to hold out).
    """
    names = list(manifest["shards"])
    if len(names) < 2:
        raise ValueError(f"need >= 2 shards to hold out a test split, got {len(names)}")
    perm = np.random.default_rng(seed).permutation(len(names))
    n_test = max(1, int(len(names) * test_frac))
    test = [names[i] for i in perm[:n_test]]
    train = [names[i] for i in perm[n_test:]]
    return train, test


def _stack_shard(
    cache_dir: Path, name: str, layers: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one shard, stacking its cached layers into ``(tok, n_layers, d_model)``."""
    with np.load(cache_dir / name) as z:
        x = np.stack([decode_act(z[f"in_{L}"]) for L in layers], axis=1)
        y = np.stack([decode_act(z[f"out_{L}"]) for L in layers], axis=1)
        kinds = z["kinds"]
    return x, y, kinds


def stream_balanced_batches(
    cache_dir: str | Path,
    shard_names: list[str],
    layers: list[int],
    *,
    batch_size: int,
    steps: int,
    buffer_shards: int = 8,
    steps_per_buffer: int = 200,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``steps`` balanced ``(x, y)`` batches by streaming shards through a shuffle buffer.

    Loads ``buffer_shards`` shards into memory at a time (reshuffled each epoch), draws
    ``steps_per_buffer`` batches from that buffer — each ~evenly split across the present token
    kinds — then evicts and loads the next shards. Memory is bounded by ``buffer_shards``; over
    many rounds the reshuffling approximates a global shuffle (tf.data-style).

    Args:
        cache_dir: the activation cache directory.
        shard_names: shard filenames to stream (e.g. the train split from :func:`split_shards`).
        layers: cached layer indices in order (the ``n_layers`` axis of ``x``/``y``).
        batch_size: tokens per batch, split across the present kinds.
        steps: number of batches to yield.
        buffer_shards: shards resident in memory at once (the memory knob).
        steps_per_buffer: batches drawn before swapping the buffer.
        seed: RNG seed.

    Yields:
        ``(x, y)`` each shaped ``(~batch_size, n_layers, d_model)``.

    Raises:
        ValueError: if ``shard_names`` is empty.
    """
    cache_dir = Path(cache_dir)
    names = list(shard_names)
    if not names:
        raise ValueError("no shards to stream from")
    layers = list(layers)
    rng = np.random.default_rng(seed)
    buffer_shards = min(buffer_shards, len(names))

    queue: list[int] = []

    def next_name() -> str:
        if not queue:  # reshuffle a fresh epoch of shard indices
            queue.extend(int(i) for i in rng.permutation(len(names)))
        return names[queue.pop()]

    def load_buffer() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        xs, ys, ks, seen = [], [], [], set()
        loaded = 0
        # Load >= buffer_shards shards. If the cache is kind-segregated (single-kind shards),
        # keep pulling until >= 2 kinds are present so batches can be balanced — bounded by the
        # shard list. Production shuffles records before caching (see scripts/b1_scale.py), so
        # shards are mixed and this stops exactly at buffer_shards.
        while loaded < len(names):
            x, y, k = _stack_shard(cache_dir, next_name(), layers)
            xs.append(x)
            ys.append(y)
            ks.append(k)
            seen.update(np.unique(k).tolist())
            loaded += 1
            if loaded >= buffer_shards and len(seen) >= 2:
                break
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        k = np.concatenate(ks, axis=0)
        pools = [p for p in (np.flatnonzero(k == kind) for kind in np.unique(k)) if p.size]
        if not pools:
            raise ValueError("shuffle buffer has no tokens")
        return x, y, pools

    emitted = 0
    while emitted < steps:
        x, y, pools = load_buffer()
        per = max(1, batch_size // len(pools))
        for _ in range(steps_per_buffer):
            if emitted >= steps:
                break
            idx = np.concatenate([rng.choice(p, size=per) for p in pools])  # balanced draw
            yield x[idx], y[idx]
            emitted += 1


def capture_stack(records: list[Any], capture_fn: Any, layers: list[int]):
    """Capture each record's activations and stack to ``(tokens, n_layers, d)`` x/y + ``kinds``.

    ``capture_fn(record) -> RecordActivations`` (e.g. ``medgemma_capture_fn``). The torch/model
    dependency lives entirely in the injected ``capture_fn``, so this stays import-light.
    """
    xs, ys, ks = [], [], []
    for rec in records:
        ra = capture_fn(rec)
        xs.append(np.stack([ra.mlp_in[L] for L in layers], axis=1))
        ys.append(np.stack([ra.mlp_out[L] for L in layers], axis=1))
        ks.append(ra.kinds)
    return np.concatenate(xs, 0), np.concatenate(ys, 0), np.concatenate(ks, 0)


def live_balanced_batches(
    records: list[Any],
    capture_fn: Any,
    layers: list[int],
    *,
    batch_size: int,
    steps: int,
    records_per_buffer: int = 64,
    steps_per_buffer: int = 100,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Like :func:`stream_balanced_batches` but fills the shuffle buffer by **capturing records
    live** (a MedGemma forward per record) instead of reading cached shards — so training data is
    bounded only by how many images/reports you have, not by disk. For a large CLT the forward is a
    small fraction of each step, so dropping the cache is worth the unbounded data.

    Captures ``records_per_buffer`` records into the buffer, draws ``steps_per_buffer`` balanced
    text/image batches, then captures the next set. ``records`` is cycled (reshuffled per epoch).
    """
    records = list(records)
    if not records:
        raise ValueError("no records to capture")
    layers = list(layers)
    rng = np.random.default_rng(seed)
    queue: list[int] = []

    def next_record() -> Any:
        if not queue:
            queue.extend(int(i) for i in rng.permutation(len(records)))
        return records[queue.pop()]

    emitted = 0
    while emitted < steps:
        x, y, k = capture_stack([next_record() for _ in range(records_per_buffer)], capture_fn,
                                layers)
        pools = [p for p in (np.flatnonzero(k == kind) for kind in np.unique(k)) if p.size]
        if not pools:
            continue
        per = max(1, batch_size // len(pools))
        for _ in range(steps_per_buffer):
            if emitted >= steps:
                break
            idx = np.concatenate([rng.choice(p, size=per) for p in pools])
            yield x[idx], y[idx]
            emitted += 1
