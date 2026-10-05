"""Training loop for the cross-layer transcoder (manifesto §Stage 1).

Ties the pipeline together: stream cached MedGemma activations (:mod:`tracecxr.transcoder.cache`)
-> train a :class:`~tracecxr.transcoder.clt.CrossLayerTranscoder` -> score with FVU split by
token kind (:mod:`tracecxr.transcoder.eval`). The lesson of A1/A3 is baked in: batches are
sampled **balanced across text and image tokens** so the dictionary covers both pathways.

The CLT operates on all cached layers at once (shape ``(batch, n_layers, d_model)``), so the
cache should hold **consecutive** layers for the cross-layer write to be meaningful. For a
small corpus we stack the activations in memory (capped by ``max_tokens``); the scale path
streams shards instead (future work — flagged inline).

Only :func:`train_clt` needs torch (via the CLT); data loading and scoring are NumPy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tracecxr.transcoder import eval as clt_eval
from tracecxr.transcoder.cache import decode_act, load_manifest
from tracecxr.transcoder.clt import CLTConfig, CrossLayerTranscoder
from tracecxr.transcoder.stream import (
    capture_stack,
    live_balanced_batches,
    split_shards,
    stream_balanced_batches,
)


def load_stacked(
    cache_dir: str | Path,
    max_tokens: int | None = None,
    seed: int = 0,
    shard_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Stack cached layers into ``(x, y, kinds)`` aligned by token position.

    Returns ``(x, y, kinds, layers)`` where ``x``/``y`` are ``(n_tokens, n_layers, d_model)``
    MLP inputs/outputs and ``kinds`` is ``(n_tokens,)`` of "text"/"image". Pass ``shard_names`` to
    load only a subset of shards (e.g. a held-out test split); defaults to the whole manifest.

    When ``max_tokens`` caps the load, shards are read in **shuffled order** so the cap samples
    across token kinds — a cache is typically written kind-ordered (text shards then image
    shards), so reading manifest order + truncating would yield a text-only, unbalanced subset.
    """
    out = Path(cache_dir)
    man = load_manifest(cache_dir)
    layers = man["layers"]
    names = list(shard_names) if shard_names is not None else list(man["shards"])
    if max_tokens is not None:
        np.random.default_rng(seed).shuffle(names)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    n = 0
    for name in names:
        with np.load(out / name) as z:
            xs.append(np.stack([decode_act(z[f"in_{L}"]) for L in layers], axis=1))  # (tok, L, d)
            ys.append(np.stack([decode_act(z[f"out_{L}"]) for L in layers], axis=1))
            kinds.append(z["kinds"])
        n += xs[-1].shape[0]
        if max_tokens is not None and n >= max_tokens:
            break
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    k = np.concatenate(kinds, axis=0)
    if max_tokens is not None and x.shape[0] > max_tokens:
        x, y, k = x[:max_tokens], y[:max_tokens], k[:max_tokens]
    return x, y, k, list(layers)


def _balanced_batches(kinds: np.ndarray, batch_size: int, steps: int, seed: int = 0):
    """Yield ``steps`` index arrays, each ~evenly split across the present token kinds."""
    rng = np.random.default_rng(seed)
    pools = {k: np.flatnonzero(kinds == k) for k in np.unique(kinds)}
    pools = {k: v for k, v in pools.items() if v.size}
    if not pools:
        raise ValueError("no training tokens to sample from (empty/too-small cache)")
    per = max(1, batch_size // len(pools))
    for _ in range(steps):
        yield np.concatenate([rng.choice(idx, size=per) for idx in pools.values()])


def _make_scorecard(clt: CrossLayerTranscoder, x_te: Any, y_te: np.ndarray,
                    k_te: np.ndarray, n_layers: int):
    """Build a held-out FVU scorecard closure (text vs image, per layer; chunked & grad-free)."""
    import torch  # noqa: PLC0415

    def scorecard(chunk: int = 8192) -> dict[str, float]:
        # Reconstruct the held-out set in chunks under inference_mode (no autograd graph) —
        # a full-set eval would spike memory on top of the resident training state.
        parts = []
        with torch.inference_mode():
            for i in range(0, x_te.shape[0], chunk):
                parts.append(clt.reconstruct(x_te[i : i + chunk]).float().cpu().numpy())
        recon = np.concatenate(parts, axis=0)  # (n_test, n_layers, d)
        per_layer = [clt_eval.fvu_by_kind(y_te[:, L], recon[:, L], k_te) for L in range(n_layers)]
        keys = per_layer[0].keys()
        return {f"fvu_{key}": float(np.nanmean([pl[key] for pl in per_layer])) for key in keys}

    return scorecard


def _run_loop(clt, batches, scorecard, *, steps, start, eval_every, ckpt, checkpoint_every,
              progress_fn=None, heartbeat_every=200):
    """Shared train/checkpoint/eval loop over an iterable of ``(x_batch, y_batch)`` GPU tensors.

    ``progress_fn(record)`` (optional) is called **as training runs** — a light heartbeat
    ``{step, recon_mse, l0}`` every ``heartbeat_every`` steps and the full ``{step, ..., fvu_*}``
    record at each eval — so callers can log incrementally (stdout / JSONL / wandb) instead of only
    at the end. The returned ``history`` is unchanged (eval records only).
    """
    history: list[dict[str, float]] = []
    last_sc: dict[str, float] = {}
    for offset, (x_b, y_b) in enumerate(batches, start=1):
        step = start + offset
        metrics = clt.train_step((x_b, y_b))
        # Checkpoint before eval so training progress is saved even if eval spikes memory.
        if ckpt and (step % checkpoint_every == 0 or step == steps):
            clt.save_checkpoint(ckpt, step)
        if step % eval_every == 0 or step == steps:
            last_sc = scorecard()
            record = {"step": step, **metrics, **last_sc}
            history.append(record)
            if progress_fn is not None:
                progress_fn(record)
        elif progress_fn is not None and heartbeat_every and step % heartbeat_every == 0:
            progress_fn({"step": step, **metrics})  # light heartbeat — no scorecard
    return history, last_sc


def _resume_step(clt: CrossLayerTranscoder, ckpt: str | None, resume: bool) -> int:
    """Restore from ``ckpt`` if it exists; return the step to resume from (0 if fresh)."""
    if ckpt and resume and Path(ckpt).exists():
        return clt.load_checkpoint(ckpt)
    return 0


def train_clt(
    cache_dir: str | Path,
    config: CLTConfig | None = None,
    *,
    steps: int = 2000,
    batch_size: int = 4096,
    eval_every: int = 500,
    test_frac: float = 0.1,
    max_tokens: int | None = None,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 500,
    resume: bool = True,
    progress_fn: Any = None,
    heartbeat_every: int = 200,
) -> dict[str, Any]:
    """Train a CLT on an **in-memory** cache and score it (FVU text vs image, per layer).

    Stacks the whole cache (capped by ``max_tokens``) and preloads it onto the device — fast for
    small corpora. For tens of millions of tokens that won't fit, use :func:`train_clt_streaming`.

    Returns ``{"clt", "history", "scorecard", "layers"}`` — the trained transcoder, the per-eval
    metric history, the final held-out scorecard, and the cached layer indices. The config's
    ``n_layers`` is set to match the cache.

    For preemptible / spot GPUs: pass ``checkpoint_dir`` to save every ``checkpoint_every`` steps;
    with ``resume=True`` a restart picks up from the last checkpoint.
    """
    from dataclasses import replace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    x, y, kinds, layers = load_stacked(cache_dir, max_tokens=max_tokens, seed=seed)
    n, n_layers, _ = x.shape

    # Copy (don't mutate) the caller's config — the CLT must match the cached layer count.
    base = config or CLTConfig(n_features=8192, k=32, span=1)
    cfg = replace(base, n_layers=n_layers)
    clt = CrossLayerTranscoder(cfg)
    dev = clt._device()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    te, tr = perm[:n_test], perm[n_test:]
    k_tr, k_te = kinds[tr], kinds[te]
    # Preload activations onto the device ONCE — copying every batch host->device per step is
    # the dominant cost for a small dictionary (the L4 bottleneck). Index resident tensors.
    # In amp mode store them as bf16 to halve the resident memory (the dominant cost at scale).
    # Only on CUDA — autocast is CUDA-only, so on CPU keep fp32 (bf16 would mismatch fp32 params).
    act_dtype = torch.bfloat16 if (cfg.amp and dev == "cuda") else torch.float32
    x_tr = torch.as_tensor(x[tr], dtype=act_dtype, device=dev)
    y_tr = torch.as_tensor(y[tr], dtype=act_dtype, device=dev)
    x_te = torch.as_tensor(x[te], dtype=act_dtype, device=dev)
    y_te = y[te]  # kept on host for NumPy FVU scoring

    scorecard = _make_scorecard(clt, x_te, y_te, k_te, n_layers)
    ckpt = str(Path(checkpoint_dir) / "clt_ckpt.pt") if checkpoint_dir else None
    start = _resume_step(clt, ckpt, resume)

    def batches():
        for idx in _balanced_batches(k_tr, batch_size, max(0, steps - start), seed + start):
            idx_t = torch.as_tensor(idx, device=dev)
            yield x_tr[idx_t], y_tr[idx_t]

    history, last_sc = _run_loop(clt, batches(), scorecard, steps=steps, start=start,
                                 eval_every=eval_every, ckpt=ckpt,
                                 checkpoint_every=checkpoint_every,
                                 progress_fn=progress_fn, heartbeat_every=heartbeat_every)
    # Reuse the final eval (loop always evals at step==steps); only compute if it never ran.
    return {"clt": clt, "history": history, "scorecard": last_sc or scorecard(), "layers": layers}


def train_clt_streaming(
    cache_dir: str | Path,
    config: CLTConfig | None = None,
    *,
    steps: int = 2000,
    batch_size: int = 4096,
    eval_every: int = 500,
    test_frac: float = 0.1,
    max_test_tokens: int | None = None,
    seed: int = 0,
    buffer_shards: int = 8,
    steps_per_buffer: int = 200,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 500,
    resume: bool = True,
    progress_fn: Any = None,
    heartbeat_every: int = 200,
) -> dict[str, Any]:
    """Train a CLT by **streaming** shards from disk — for caches too large to hold in memory.

    Same contract as :func:`train_clt`, but the training batches come from
    :func:`tracecxr.transcoder.stream.stream_balanced_batches` (a bounded shuffle buffer) and the
    held-out set is a disjoint **shard split** (:func:`~tracecxr.transcoder.stream.split_shards`)
    small enough to keep resident for scoring. ``buffer_shards`` bounds memory; ``steps_per_buffer``
    sets how many batches are drawn before swapping the buffer. Checkpoint/resume is identical.

    ``max_test_tokens`` **caps the held-out set** (sampled across kinds via ``load_stacked``'s
    shuffle): without it, ``test_frac`` of a multi-million-token cache would be hundreds of
    thousands of tokens — ~100s of GB on the host and tens of GB resident on the GPU (on top of a
    large CLT) — an OOM at startup. A few x10k tokens give a stable FVU; pass it for any real run.
    """
    from dataclasses import replace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    man = load_manifest(cache_dir)
    layers = list(man["layers"])
    train_names, test_names = split_shards(man, test_frac=test_frac, seed=seed)
    x_te_np, y_te, k_te, _ = load_stacked(cache_dir, shard_names=test_names,
                                          max_tokens=max_test_tokens, seed=seed)
    n_layers = len(layers)

    base = config or CLTConfig(n_features=8192, k=32, span=1)
    cfg = replace(base, n_layers=n_layers)
    clt = CrossLayerTranscoder(cfg)
    dev = clt._device()
    act_dtype = torch.bfloat16 if (cfg.amp and dev == "cuda") else torch.float32
    x_te = torch.as_tensor(x_te_np, dtype=act_dtype, device=dev)

    scorecard = _make_scorecard(clt, x_te, y_te, k_te, n_layers)
    ckpt = str(Path(checkpoint_dir) / "clt_ckpt.pt") if checkpoint_dir else None
    start = _resume_step(clt, ckpt, resume)

    def batches():
        stream = stream_balanced_batches(
            cache_dir, train_names, layers, batch_size=batch_size, steps=max(0, steps - start),
            buffer_shards=buffer_shards, steps_per_buffer=steps_per_buffer, seed=seed + start,
        )
        for x_np, y_np in stream:
            yield (torch.as_tensor(x_np, dtype=act_dtype, device=dev),
                   torch.as_tensor(y_np, dtype=act_dtype, device=dev))

    history, last_sc = _run_loop(clt, batches(), scorecard, steps=steps, start=start,
                                 eval_every=eval_every, ckpt=ckpt,
                                 checkpoint_every=checkpoint_every,
                                 progress_fn=progress_fn, heartbeat_every=heartbeat_every)
    return {"clt": clt, "history": history, "scorecard": last_sc or scorecard(), "layers": layers}


def train_clt_onthefly(
    records: list[Any],
    capture_fn: Any,
    eval_records: list[Any],
    layers: list[int],
    config: CLTConfig | None = None,
    *,
    steps: int = 2000,
    batch_size: int = 1024,
    eval_every: int = 500,
    records_per_buffer: int = 64,
    steps_per_buffer: int = 100,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 500,
    resume: bool = True,
    progress_fn: Any = None,
    heartbeat_every: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a CLT by capturing activations **live** (no disk cache) — data is bounded only by the
    image/report corpus, not by disk. This is the path to feeding a large CLT enough data: a cached
    run caps at the fast disk (~2 M tokens), but for a multi-billion-param CLT a MedGemma forward is
    only ~6% of a step, so re-capturing each batch is cheap relative to the disk straitjacket.

    Same return contract as :func:`train_clt_streaming`. Training batches come from
    :func:`~tracecxr.transcoder.stream.live_balanced_batches` (MedGemma forward per record, balanced
    text/image), and the held-out set is ``eval_records`` — captured once and kept resident for FVU.

    MedGemma (inside ``capture_fn``) is resident **alongside** the CLT on the GPU, so size the CLT
    batch for the combined memory (the model adds ~8 GB on top of the CLT's training footprint).
    """
    from dataclasses import replace  # noqa: PLC0415

    import torch  # noqa: PLC0415

    layers = list(layers)
    n_layers = len(layers)
    base = config or CLTConfig(n_features=8192, k=32, span=1)
    cfg = replace(base, n_layers=n_layers)
    clt = CrossLayerTranscoder(cfg)
    dev = clt._device()
    act_dtype = torch.bfloat16 if (cfg.amp and dev == "cuda") else torch.float32

    # held-out set: capture once, keep resident for FVU scoring (disjoint from the training pool).
    x_te_np, y_te, k_te = capture_stack(list(eval_records), capture_fn, layers)
    x_te = torch.as_tensor(x_te_np, dtype=act_dtype, device=dev)
    scorecard = _make_scorecard(clt, x_te, y_te, k_te, n_layers)

    ckpt = str(Path(checkpoint_dir) / "clt_ckpt.pt") if checkpoint_dir else None
    start = _resume_step(clt, ckpt, resume)

    def batches():
        gen = live_balanced_batches(
            records, capture_fn, layers, batch_size=batch_size, steps=max(0, steps - start),
            records_per_buffer=records_per_buffer, steps_per_buffer=steps_per_buffer,
            seed=seed + start,
        )
        for x_np, y_np in gen:
            yield (torch.as_tensor(x_np, dtype=act_dtype, device=dev),
                   torch.as_tensor(y_np, dtype=act_dtype, device=dev))

    history, last_sc = _run_loop(clt, batches(), scorecard, steps=steps, start=start,
                                 eval_every=eval_every, ckpt=ckpt,
                                 checkpoint_every=checkpoint_every,
                                 progress_fn=progress_fn, heartbeat_every=heartbeat_every)
    return {"clt": clt, "history": history, "scorecard": last_sc or scorecard(), "layers": layers}
