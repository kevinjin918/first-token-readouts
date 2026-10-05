"""Export a CLT training checkpoint to a weights-only safetensors file, and check the export.

A training checkpoint (:meth:`CrossLayerTranscoder.save_checkpoint`, e.g. the released
``clt_ckpt.pt``) is a pickle holding the parameters, the optimizer state and the step. Inference
needs only the parameters, and only part of the decoder: ``W_dec_<o>[l]`` writes source layer
``l``'s features into layer ``l + o``, so rows ``l >= n_layers - o`` would write past the last layer
and are never read (:meth:`CrossLayerTranscoder.decode` slices them off). The export keeps
``W_enc``, ``b_enc``, ``b_dec`` and the used decoder rows, and stores them without pickle.

The writer streams one tensor at a time, so exporting the 6.24B-parameter checkpoint needs about
one tensor (~0.7 GB) of memory rather than the whole model: the source is memory-mapped and the
safetensors file is written by hand (an 8-byte header length, a JSON header, then the raw
little-endian bytes in header order), which ``safetensors.safe_open`` reads back.

torch and safetensors are imported lazily, so importing this module stays offline.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

_DTYPES = {"float32": "F32", "bfloat16": "BF16", "float16": "F16"}


def _load_params(ckpt_path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Memory-map a training checkpoint; return its parameters and the metadata to carry over."""
    import torch  # noqa: PLC0415

    # weights_only=False: the checkpoint also pickles optimizer state. Only load files you trust
    # (the released one has a published MD5).
    ckpt = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
    params = ckpt["params"]
    meta = {"format": "pt", "d_model": str(int(ckpt["d_model"])), "step": str(int(ckpt["step"])),
            "n_layers": str(int(params["W_enc"].shape[0])),
            "n_features": str(int(params["W_enc"].shape[2])),
            "span": str(sum(1 for k in params if k.startswith("W_dec_")) - 1)}
    return params, meta


def _kept(params: dict[str, Any]) -> dict[str, Any]:
    """The tensors inference reads: W_enc, both biases, and each decoder block's used rows."""
    n_layers = int(params["W_enc"].shape[0])
    out = {k: params[k] for k in ("W_enc", "b_enc", "b_dec")}
    offsets = sorted(int(k.removeprefix("W_dec_")) for k in params if k.startswith("W_dec_"))
    for o in offsets:
        out[f"W_dec_{o}"] = params[f"W_dec_{o}"][: n_layers - o]
    return out


def export_safetensors(ckpt_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Write the kept tensors of ``ckpt_path`` to ``out_path``; return a summary of the export."""
    import torch  # noqa: PLC0415

    params, meta = _load_params(ckpt_path)
    tensors = _kept(params)
    header: dict[str, Any] = {"__metadata__": meta}
    start = 0
    for name, t in tensors.items():
        dtype = _DTYPES[str(t.dtype).removeprefix("torch.")]
        nbytes = t.numel() * t.element_size()
        header[name] = {"dtype": dtype, "shape": list(t.shape),
                        "data_offsets": [start, start + nbytes]}
        start += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # pad the header so the data starts 8-byte aligned
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for t in tensors.values():
            # a leading-dimension slice of a contiguous tensor is contiguous; view as bytes
            f.write(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    tmp.replace(out_path)
    n_params = sum(t.numel() for t in tensors.values())
    return {"n_params": n_params, "bytes": out_path.stat().st_size, **meta}


def verify_export(ckpt_path: str | Path, out_path: str | Path) -> int:
    """Check every tensor in ``out_path`` is bitwise equal to the kept part of ``ckpt_path``.

    Reads the export with the official ``safetensors`` reader, so this also checks the file
    format. Returns the number of tensors compared; raises ``AssertionError`` on any mismatch.
    """
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    params, meta = _load_params(ckpt_path)
    want = _kept(params)
    with safe_open(str(out_path), framework="pt", device="cpu") as f:
        assert f.metadata() == meta, f"metadata {f.metadata()} != {meta}"
        assert set(f.keys()) == set(want), f"keys differ: {sorted(set(f.keys()) ^ set(want))}"
        for name, ref in want.items():
            got = f.get_tensor(name)
            assert got.dtype == ref.dtype and got.shape == ref.shape, name
            assert torch.equal(got, ref), f"{name} differs from the checkpoint"
    return len(want)
