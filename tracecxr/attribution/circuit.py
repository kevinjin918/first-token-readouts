"""Read a circuit out of an attribution graph (manifesto §Stage 2).

Pure graph analysis — no model needed, fully CI-tested. The key operation is ranking the CLT
feature nodes by how strongly they drive the finding logit, split by where they live: the
**report-prior** features sit at prompt/context (text) positions and are the ones that carry
the finding without pixels, while the **image-evidence** features sit at image-patch
positions. :func:`report_prior_features` returns the suppression targets to hand to the
clamp (:class:`~tracecxr.intervention.FeatureEdit`); the clamp-vs-no-clamp drop on those is
the causal test.
"""

from __future__ import annotations

from dataclasses import dataclass

from tracecxr.attribution.graph import AttributionGraph, NodeType


@dataclass(frozen=True)
class FeatureScore:
    """A CLT feature and its signed total attribution to the output logit(s).

    Attributes:
        layer: model decoder layer the feature lives at.
        feature: feature index within that layer's dictionary.
        attribution: summed signed edge weight from this feature into the output nodes
            (positive = pushes the finding up).
        position: token / image-patch position the feature fired at.
        node_id: the feature node's id in the graph.
    """

    layer: int
    feature: int
    attribution: float
    position: int | None
    node_id: str


def features_by_attribution(
    graph: AttributionGraph,
    *,
    kind: str = "all",
    top_k: int | None = None,
) -> list[FeatureScore]:
    """Rank feature nodes by their signed attribution to the output (logit) nodes.

    Args:
        graph: an attribution graph (raw or pruned) with feature→output edges.
        kind: ``"all"``, ``"text"`` (features NOT at an image-patch position — the
            report-prior side), or ``"image"`` (features at an image-patch position — the
            image-evidence side). Position kind is inferred from the graph's
            :attr:`NodeType.INPUT_IMAGE` nodes.
        top_k: keep only the highest-attribution ``top_k`` (most positive first); ``None``
            keeps all.

    Returns:
        :class:`FeatureScore` rows, most positive attribution first.
    """
    if kind not in ("all", "text", "image"):
        raise ValueError(f"kind must be 'all', 'text', or 'image', got {kind!r}")
    image_positions = {n.position for n in graph.nodes_by_type(NodeType.INPUT_IMAGE)}
    output_ids = {n.id for n in graph.nodes_by_type(NodeType.OUTPUT)}

    summed: dict[str, float] = {}
    for e in graph.edges:
        if e.target in output_ids:
            node = graph.nodes.get(e.source)
            if node is not None and node.node_type == NodeType.FEATURE:
                summed[e.source] = summed.get(e.source, 0.0) + e.weight

    rows: list[FeatureScore] = []
    for fid, attribution in summed.items():
        node = graph.nodes[fid]
        is_image = node.position in image_positions
        if kind == "image" and not is_image:
            continue
        if kind == "text" and is_image:
            continue
        rows.append(
            FeatureScore(
                layer=int(node.metadata.get("layer", -1)),
                feature=int(node.metadata.get("feature", -1)),
                attribution=attribution,
                position=node.position,
                node_id=fid,
            )
        )
    rows.sort(key=lambda r: r.attribution, reverse=True)
    return rows[:top_k] if top_k is not None else rows


def report_prior_features(graph: AttributionGraph, *, top_k: int = 8) -> list[FeatureScore]:
    """The top text-position features driving the finding — the suppression targets.

    Convenience wrapper for ``features_by_attribution(graph, kind="text", top_k=top_k)``.
    Run it on the **no-image** graph: those features carry the finding with no pixels, so
    they are the report-prior circuit to clamp.
    """
    return features_by_attribution(graph, kind="text", top_k=top_k)
