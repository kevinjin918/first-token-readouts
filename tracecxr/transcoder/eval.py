"""Reconstruction-fidelity metrics for transcoders / CLTs (manifesto §Stage 1).

The whole Stage 1 question is "does the sparse replacement faithfully mimic MedGemma's MLPs,
*including* on image tokens" (see ``results/a1_fidelity.md`` and ``results/a3_imageaware.md``).
This module is the architecture-agnostic scoring used by both the A1/A3 prototypes and the
real CLT training: **FVU** (fraction of variance unexplained) split by token kind, plus the
sparsity (**L0**) the manifesto targets in the tens-to-low-hundreds.

Pure NumPy — no torch — so it runs in CI and against either backend's arrays.

FVU(true, recon) = ||true - recon||^2 / ||true - mean(true)||^2 over the selected positions.
  * FVU = 0  -> perfect reconstruction
  * FVU = 1  -> no better than predicting the mean activation
  * FVU > 1  -> worse than the mean (the image-token OOD signature from A1)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Token-kind labels used to split fidelity (the A1/A3 text-vs-image comparison).
TEXT = "text"
IMAGE = "image"


def fvu(true: np.ndarray, recon: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of variance unexplained over the (optionally masked) positions.

    Args:
        true: ground-truth activations, shape ``(n_positions, d_model)``.
        recon: reconstructed activations, same shape.
        mask: optional boolean array ``(n_positions,)`` selecting positions; None = all.

    Returns:
        FVU as a float; ``nan`` if no positions are selected.
    """
    t = np.asarray(true, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError(f"shape mismatch: true {t.shape} vs recon {r.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        t, r = t[m], r[m]
    if t.shape[0] == 0:
        return float("nan")
    num = float(((t - r) ** 2).sum())
    den = float(((t - t.mean(axis=0, keepdims=True)) ** 2).sum())
    return num / den if den > 0 else float("nan")


def fvu_by_kind(
    true: np.ndarray, recon: np.ndarray, kinds: Sequence[str]
) -> dict[str, float]:
    """FVU computed separately for each token kind (e.g. ``text`` vs ``image``).

    Args:
        true: ground-truth activations ``(n_positions, d_model)``.
        recon: reconstruction, same shape.
        kinds: per-position kind labels, length ``n_positions``.

    Returns:
        ``{kind: fvu}`` for every distinct kind present, plus ``"all"`` over all positions.
    """
    kinds_arr = np.asarray(kinds)
    out: dict[str, float] = {"all": fvu(true, recon)}
    for kind in sorted(set(kinds_arr.tolist())):
        out[str(kind)] = fvu(true, recon, mask=(kinds_arr == kind))
    return out


def l0(feature_acts: np.ndarray, eps: float = 1e-6) -> float:
    """Mean number of active features per token (the sparsity the manifesto targets).

    Args:
        feature_acts: feature activations ``(n_positions, n_features)`` (or with leading
            layer axes — everything but the last axis is treated as a position).
        eps: magnitude above which a feature counts as active.

    Returns:
        Mean count of ``|activation| > eps`` per position.
    """
    a = np.asarray(feature_acts)
    flat = a.reshape(-1, a.shape[-1])
    return float((np.abs(flat) > eps).sum(axis=-1).mean())


def summarize(
    true: np.ndarray,
    recon: np.ndarray,
    kinds: Sequence[str],
    feature_acts: np.ndarray | None = None,
) -> dict[str, float]:
    """One call producing the Stage 1 scorecard: per-kind FVU (+ optional L0).

    Returns a flat dict like ``{"fvu_all", "fvu_text", "fvu_image", "l0"}`` — the row the
    training loop logs and the eval reports (cf. ``results/a3_imageaware.json``).
    """
    out = {f"fvu_{k}": v for k, v in fvu_by_kind(true, recon, kinds).items()}
    if feature_acts is not None:
        out["l0"] = l0(feature_acts)
    return out
