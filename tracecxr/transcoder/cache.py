"""Activation caching for CLT training (manifesto §Stage 1).

The CLT is trained to reconstruct MedGemma's MLP outputs from its inputs, on **CXR-conditioned
activations** (image *and* text tokens — the lesson of ``results/a1_fidelity.md`` /
``a3_imageaware.md``). Running the 4B model is the expensive part, so we run it once over the
corpus and cache the per-layer ``(mlp_in, mlp_out)`` activations plus a per-token kind label
to sharded ``.npz`` on disk; training then streams from disk without touching the model.

The orchestration (sharding, manifest, round-trip) is decoupled from the model via an
injectable ``capture_fn``, so it is fully testable with a mock — the real torch/transformers
capture path (:func:`medgemma_capture_fn`) is the only piece needing a GPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


#: One record's captured activations. Arrays are ``(seq, d_model)`` per layer; ``kinds`` is
#: ``(seq,)`` of "text"/"image" labels (see :mod:`tracecxr.transcoder.eval`).
@dataclass
class RecordActivations:
    record_id: str
    mlp_in: dict[int, np.ndarray]
    mlp_out: dict[int, np.ndarray]
    kinds: np.ndarray
    #: Model token id per kept position (aligned with ``kinds``). Optional/back-compat:
    #: populated by ``medgemma_capture_fn`` so text-position tokens can be rendered for
    #: feature dashboards; None for readers/producers that don't need it.
    token_ids: np.ndarray | None = None


#: Signature of a capture function: one record -> its per-layer activations.
CaptureFn = Callable[[Any], RecordActivations]


def decode_act(arr: np.ndarray) -> np.ndarray:
    """Decode a cached activation array to float32.

    A **bf16 cache** stores each activation's bf16 *bit pattern* as ``int16`` — half the disk of
    fp32 and lossless relative to the bf16-trained CLT (bf16 is the high 16 bits of fp32, so the
    round-trip is exact and there is no fp16-style overflow). An ``int16`` array is reinterpreted
    back to fp32 here; an already-fp32 array passes through. Readers call this on ``in_``/``out_``
    arrays (never on ``kinds``).
    """
    if arr.dtype == np.int16:
        bits = arr.view(np.uint16).astype(np.uint32) << 16  # bf16 -> high 16 bits of fp32
        return bits.view(np.float32)
    return np.asarray(arr, dtype=np.float32)


def cache_corpus(
    records: Iterable[Any],
    layers: list[int],
    out_dir: str | Path,
    capture_fn: CaptureFn,
    shard_size: int = 64,
    resume: bool = True,
) -> dict[str, Any]:
    """Run ``capture_fn`` over ``records`` and write sharded activations + a manifest.

    Each shard concatenates ``shard_size`` records' tokens into per-layer ``(N, d_model)``
    arrays (``in_<layer>`` / ``out_<layer>``) plus a ``kinds`` array, saved as ``shard_*.npz``.
    A ``manifest.json`` records layers, shard files, and total token counts by kind.

    **Spot-safe resume:** with ``resume=True`` (default), already-written ``shard_*.npz`` files are
    kept and the matching number of leading records (``n_existing_shards × shard_size``) are
    skipped, so a preempted caching run continues instead of starting over (a shard is only ever
    written on a full flush, so existing shards are always complete). Pass the *same* records list.

    Returns the manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in out.glob("shard_*.npz")) if resume else []
    shards: list[str] = list(existing)
    kind_counts: dict[str, int] = {}
    # recount kinds from already-cached shards so the manifest totals stay correct on resume.
    for name in existing:
        with np.load(out / name) as z:
            uniq, counts = np.unique(z["kinds"], return_counts=True)
            for k, c in zip(uniq, counts, strict=True):
                kind_counts[str(k)] = kind_counts.get(str(k), 0) + int(c)
    n_skip = len(existing) * shard_size
    batch: list[RecordActivations] = []
    n_records = n_skip

    def flush(idx: int) -> None:
        if not batch:
            return
        payload: dict[str, np.ndarray] = {}
        for L in layers:
            payload[f"in_{L}"] = np.concatenate([r.mlp_in[L] for r in batch], axis=0)
            payload[f"out_{L}"] = np.concatenate([r.mlp_out[L] for r in batch], axis=0)
        payload["kinds"] = np.concatenate([r.kinds for r in batch], axis=0)
        name = f"shard_{idx:05d}.npz"
        np.savez(out / name, **payload)
        shards.append(name)
        uniq, counts = np.unique(payload["kinds"], return_counts=True)
        for k, c in zip(uniq, counts, strict=True):
            kind_counts[str(k)] = kind_counts.get(str(k), 0) + int(c)

    for i, rec in enumerate(records):
        if i < n_skip:  # already cached on a previous (preempted) run
            continue
        batch.append(capture_fn(rec))
        n_records += 1
        if len(batch) >= shard_size:
            flush(len(shards))
            batch = []
    flush(len(shards))

    manifest = {
        "layers": layers,
        "shards": shards,
        "n_records": n_records,
        "shard_size": shard_size,
        "token_counts": kind_counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(out_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "manifest.json").read_text())


def iter_shards(out_dir: str | Path) -> Iterator[dict[str, np.ndarray]]:
    """Yield each shard's arrays (``in_<L>``, ``out_<L>``, ``kinds``) in manifest order."""
    out = Path(out_dir)
    for name in load_manifest(out)["shards"]:
        with np.load(out / name) as z:
            yield {k: z[k] for k in z.files}


def load_layer(out_dir: str | Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate one layer's ``(mlp_in, mlp_out, kinds)`` across all shards.

    Convenience for small corpora / eval; for large training streams use :func:`iter_shards`.
    """
    ins, outs, kinds = [], [], []
    for shard in iter_shards(out_dir):
        ins.append(decode_act(shard[f"in_{layer}"]))
        outs.append(decode_act(shard[f"out_{layer}"]))
        kinds.append(shard["kinds"])
    return (
        np.concatenate(ins, axis=0),
        np.concatenate(outs, axis=0),
        np.concatenate(kinds, axis=0),
    )


def medgemma_capture_fn(
    layers: list[int],
    *,
    model_id: str | None = None,
    prompt: str = "Interpret this chest X-ray.",
    max_text_tokens: int = 96,
    store_bf16: bool = False,
) -> CaptureFn:
    """Build the real capture function over MedGemma (lazy — imports torch on first call).

    For a record with an image, captures the image-token positions (kind "image") and the
    prompt's text positions (kind "text"); for a text-only record, all positions are "text".
    Hooks each ``model.language_model.layers[L].mlp`` for its input and output. Requires a GPU
    and model weights — covered by ``requires_gpu`` tests only.

    With ``store_bf16`` the activations are stored as the bf16 bit pattern (``int16``) — half the
    disk, lossless vs the bf16-trained CLT (see :func:`decode_act`, which readers use to restore
    fp32). The fp32 default is kept for back-compat / small caches.
    """
    state: dict[str, Any] = {}

    def _load() -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        from tracecxr.core.config import MODELS

        mid = model_id or MODELS["medgemma"].locator
        proc = AutoProcessor.from_pretrained(mid)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype).to(dev).eval()
        state.update(
            torch=torch, proc=proc, model=model, dev=dev,
            img_id=int(model.config.image_token_index),
            layers_mod=model.model.language_model.layers,
        )

    def capture(record: Any) -> RecordActivations:
        if not state:
            _load()
        torch = state["torch"]
        cap: dict[int, tuple[Any, Any]] = {}

        def mk(i: int):
            def hook(_m, inp, o):  # noqa: ANN001
                oo = o[0] if isinstance(o, tuple) else o
                cap[i] = (inp[0].detach()[0], oo.detach()[0])
            return hook

        handles = [state["layers_mod"][i].mlp.register_forward_hook(mk(i)) for i in layers]
        try:
            image = getattr(record, "image", None)
            # Text records carry their own text (e.g. a report); image records use ``prompt``.
            text = getattr(record, "text", None) or prompt
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            if image is not None:
                from PIL import Image  # noqa: PLC0415

                pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
                content.insert(0, {"type": "image", "image": pil.convert("RGB")})
            inputs = state["proc"].apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to(state["dev"])
            with torch.inference_mode():
                state["model"](**inputs)
            ids = inputs["input_ids"][0]
            img_mask = (ids == state["img_id"]).cpu().numpy()
            kinds = np.where(img_mask, "image", "text")
            # cap on text-only sequences to keep shards balanced.
            keep = np.ones(ids.shape[0], dtype=bool)
            if image is None and ids.shape[0] > max_text_tokens:
                keep[max_text_tokens:] = False
            kinds = kinds[keep]

            def _store(t: Any) -> np.ndarray:  # noqa: ANN401
                if store_bf16:  # bf16 bits as int16 -> half the disk, lossless vs the bf16 CLT
                    return t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy()
                return t.float().cpu().numpy()

            mlp_in = {L: _store(cap[L][0])[keep] for L in layers}
            mlp_out = {L: _store(cap[L][1])[keep] for L in layers}
            tok_ids = ids.cpu().numpy()[keep]
        finally:
            for h in handles:
                h.remove()
        return RecordActivations(
            record_id=str(getattr(record, "id", "rec")),
            mlp_in=mlp_in, mlp_out=mlp_out, kinds=kinds, token_ids=tok_ids,
        )

    capture.state = state  # expose loaded proc/model/tokenizer to callers (populated on 1st call)
    return capture
