"""Stage 1 cross-layer transcoder (CLT) — the interpretable replacement model.

Per the TraceCXR manifesto (§"Stage 1: build the interpretable replacement model (the
cross-layer transcoder)"), this package scaffolds the CLT that replaces MedGemma's decoder
MLPs with a bank of sparse features. The cross-layer property — each feature reads from the
residual stream at its own layer and writes into the reconstruction of all later MLP layers —
collapses multi-layer computation into single nameable features and shortens attribution-graph
paths (the paper's average path length drops from 3.7 with per-layer transcoders to 2.3 with
CLTs), which is what makes the downstream graphs (Stage 2) legible.

This is an interface stub: importing it pulls in no heavy dependencies and runs no training.
"""

from __future__ import annotations

from tracecxr.transcoder.clt import CLTConfig, CrossLayerTranscoder

__all__ = ["CLTConfig", "CrossLayerTranscoder"]
