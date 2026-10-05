"""Stage 2 attribution graphs (manifesto §Stage 2).

Usable data structures (:class:`AttributionGraph`, :class:`Node`, :class:`Edge`,
:class:`NodeType`), graph pruning (:func:`prune`, implemented), and graph assembly
(:func:`build_attribution_graph`) from an injectable :class:`AttributionBackend` — the
production backend (MedGemma + the trained CLT, GPU-only) lands as its own module.
"""

from __future__ import annotations

from tracecxr.attribution.circuit import (
    FeatureScore,
    features_by_attribution,
    report_prior_features,
)
from tracecxr.attribution.graph import (
    AttributionBackend,
    AttributionGraph,
    Edge,
    Node,
    NodeType,
    RawAttribution,
    build_attribution_graph,
    error_id,
    feature_id,
    input_image_id,
    input_token_id,
    output_id,
    prune,
)

__all__ = [
    "AttributionBackend",
    "AttributionGraph",
    "Edge",
    "FeatureScore",
    "Node",
    "NodeType",
    "RawAttribution",
    "build_attribution_graph",
    "error_id",
    "feature_id",
    "features_by_attribution",
    "input_image_id",
    "input_token_id",
    "output_id",
    "prune",
    "report_prior_features",
]
