"""Tests for the Stage 2 attribution-graph interface (manifesto §Stage 2).

The data structures, :func:`prune`, and the graph assembly in
:func:`build_attribution_graph` are real and exercised here against a deterministic mock
backend; the production (GPU) backend is a separate torch-only module.
"""

from __future__ import annotations

import pytest
from tracecxr.attribution import (
    AttributionGraph,
    Edge,
    Node,
    NodeType,
    RawAttribution,
    build_attribution_graph,
    feature_id,
    input_image_id,
    input_token_id,
    output_id,
    prune,
)


class _MockBackend:
    """A deterministic backend yielding the two-path graph (image vs report-prior).

    Satisfies the :class:`~tracecxr.attribution.AttributionBackend` protocol structurally.
    """

    def __init__(self, *, with_image: bool = True) -> None:
        self.with_image = with_image
        self.calls: list[dict] = []

    def trace(self, *, prompt, finding_token, image=None, n_logits=10, logit_mass=0.95):
        self.calls.append({"prompt": prompt, "finding_token": finding_token, "image": image})
        out = output_id(5, finding_token)
        ff = feature_id(14, 5, 7)  # finding feature
        prior = feature_id(10, 4, 3)  # report-prior feature
        tok = input_token_id(4)
        err = "err:L14:p5"
        nodes = [
            Node(out, NodeType.OUTPUT, label=finding_token, activation=2.0, position=5),
            Node(ff, NodeType.FEATURE, label="finding", position=5, metadata={"layer": 14}),
            Node(prior, NodeType.FEATURE, label="report-prior", position=4, metadata={"layer": 10}),
            Node(tok, NodeType.INPUT_TOKEN, label="effusion?", position=4),
            Node(err, NodeType.ERROR, label="residual@5", position=5, metadata={"layer": 14}),
        ]
        edges = [
            (tok, prior, 1.1),
            (prior, ff, 0.6),
            (ff, out, 1.5),
            (err, out, -0.2),
        ]
        if image is not None:  # the image-evidence path only exists when an image is present
            visual = feature_id(8, 2, 1)
            patch = input_image_id(2)
            nodes.append(Node(visual, NodeType.FEATURE, label="visual", position=2,
                              metadata={"layer": 8}))
            nodes.append(Node(patch, NodeType.INPUT_IMAGE, label="patch[2]", position=2))
            edges.append((patch, visual, 0.9))
            edges.append((visual, ff, 0.7))
        return RawAttribution(nodes=nodes, edges=edges, metadata={"replacement_score": 0.83})


def _two_path_graph() -> AttributionGraph:
    """A tiny hand-built graph: an image-evidence path and a report-prior path
    competing into the 'effusion' output node (manifesto §Stage 2)."""
    g = AttributionGraph(metadata={"finding_token": "effusion"})
    g.add_node(Node("out", NodeType.OUTPUT, label="effusion", activation=2.0, position=5))
    g.add_node(Node("feat_finding", NodeType.FEATURE, label="effusion-feature", position=5))
    g.add_node(Node("feat_visual", NodeType.FEATURE, label="low-level-visual", position=2))
    g.add_node(Node("feat_prior", NodeType.FEATURE, label="report-prior", position=4))
    g.add_node(Node("img_patch", NodeType.INPUT_IMAGE, label="patch[2]", position=2))
    g.add_node(Node("tok_prompt", NodeType.INPUT_TOKEN, label="effusion?", position=4))
    g.add_node(Node("err", NodeType.ERROR, label="residual@5", position=5))

    # image-evidence path
    g.add_edge("img_patch", "feat_visual", 0.9)
    g.add_edge("feat_visual", "feat_finding", 0.7)
    # report-prior path
    g.add_edge("tok_prompt", "feat_prior", 1.1)
    g.add_edge("feat_prior", "feat_finding", 0.6)
    # into the logit, plus a (negative) error contribution
    g.add_edge("feat_finding", "out", 1.5)
    g.add_edge("err", "out", -0.2)
    return g


def test_add_node_returns_and_stores() -> None:
    g = AttributionGraph()
    node = g.add_node(Node("a", NodeType.OUTPUT, label="x"))
    assert node.id == "a"
    assert g.nodes["a"] is node


def test_add_duplicate_node_raises() -> None:
    g = AttributionGraph()
    g.add_node(Node("a", NodeType.FEATURE))
    with pytest.raises(ValueError, match="duplicate node id"):
        g.add_node(Node("a", NodeType.FEATURE))


def test_add_edge_requires_known_endpoints() -> None:
    g = AttributionGraph()
    g.add_node(Node("a", NodeType.FEATURE))
    with pytest.raises(KeyError, match="unknown target"):
        g.add_edge("a", "missing", 1.0)
    with pytest.raises(KeyError, match="unknown source"):
        g.add_edge("missing", "a", 1.0)


def test_add_edge_returns_edge() -> None:
    g = AttributionGraph()
    g.add_node(Node("a", NodeType.FEATURE))
    g.add_node(Node("b", NodeType.OUTPUT))
    edge = g.add_edge("a", "b", -0.3)
    assert isinstance(edge, Edge)
    assert (edge.source, edge.target, edge.weight) == ("a", "b", -0.3)
    assert g.edges == [edge]


def test_structure() -> None:
    g = _two_path_graph()
    assert len(g.nodes) == 7
    assert len(g.edges) == 6
    # every edge endpoint is a real node
    for e in g.edges:
        assert e.source in g.nodes
        assert e.target in g.nodes


def test_nodes_by_type() -> None:
    g = _two_path_graph()
    assert [n.id for n in g.nodes_by_type(NodeType.OUTPUT)] == ["out"]
    assert [n.id for n in g.nodes_by_type(NodeType.INPUT_IMAGE)] == ["img_patch"]
    assert [n.id for n in g.nodes_by_type(NodeType.INPUT_TOKEN)] == ["tok_prompt"]
    assert [n.id for n in g.nodes_by_type(NodeType.ERROR)] == ["err"]
    feature_ids = {n.id for n in g.nodes_by_type(NodeType.FEATURE)}
    assert feature_ids == {"feat_finding", "feat_visual", "feat_prior"}


def test_nodes_by_type_preserves_insertion_order() -> None:
    g = _two_path_graph()
    assert [n.id for n in g.nodes_by_type(NodeType.FEATURE)] == [
        "feat_finding",
        "feat_visual",
        "feat_prior",
    ]


def test_to_dict_shape() -> None:
    g = _two_path_graph()
    d = g.to_dict()
    assert set(d) == {"nodes", "edges", "metadata"}
    assert len(d["nodes"]) == 7
    assert len(d["edges"]) == 6
    assert d["metadata"]["finding_token"] == "effusion"
    out = next(n for n in d["nodes"] if n["id"] == "out")
    assert out["node_type"] == "output"
    assert out["activation"] == 2.0
    assert out["position"] == 5


def test_to_dict_round_trip() -> None:
    g = _two_path_graph()
    restored = AttributionGraph.from_dict(g.to_dict())
    assert restored.to_dict() == g.to_dict()
    assert {n.id for n in restored.nodes_by_type(NodeType.FEATURE)} == {
        "feat_finding",
        "feat_visual",
        "feat_prior",
    }
    restored_edge = restored.edges[-1]
    assert (restored_edge.source, restored_edge.target, restored_edge.weight) == (
        "err",
        "out",
        -0.2,
    )


def test_build_attribution_graph_requires_backend() -> None:
    with pytest.raises(NotImplementedError, match="needs an AttributionBackend"):
        build_attribution_graph(prompt="<image> is there an effusion?", finding_token="effusion")


def test_build_attribution_graph_assembles_from_backend() -> None:
    backend = _MockBackend(with_image=True)
    g = build_attribution_graph(
        prompt="<image> is there an effusion?",
        finding_token="effusion",
        backend=backend,
        image=object(),  # any non-None stands in for a CXR
    )
    # the output node and both pathways are present
    assert g.nodes_by_type(NodeType.OUTPUT)[0].label == "effusion"
    assert g.nodes_by_type(NodeType.INPUT_IMAGE)  # image-evidence entry
    assert g.nodes_by_type(NodeType.INPUT_TOKEN)  # report-prior entry
    assert g.nodes_by_type(NodeType.ERROR)
    # every edge endpoint resolved to a declared node
    for e in g.edges:
        assert e.source in g.nodes and e.target in g.nodes
    # case + backend metadata recorded
    assert g.metadata["finding_token"] == "effusion"
    assert g.metadata["has_image"] is True
    assert g.metadata["replacement_score"] == 0.83
    assert backend.calls[0]["finding_token"] == "effusion"


def test_build_attribution_graph_no_image_drops_image_path() -> None:
    g = build_attribution_graph(
        prompt="is there an effusion?",
        finding_token="effusion",
        backend=_MockBackend(),
        image=None,
    )
    assert g.metadata["has_image"] is False
    assert g.nodes_by_type(NodeType.INPUT_IMAGE) == []  # no image -> no image-evidence path
    assert g.nodes_by_type(NodeType.INPUT_TOKEN)  # report-prior path still drives the finding


def test_build_attribution_graph_rejects_dangling_edge() -> None:
    class _BadBackend:
        def trace(self, *, prompt, finding_token, image=None, n_logits=10, logit_mass=0.95):
            out = output_id(0, finding_token)
            return RawAttribution(
                nodes=[Node(out, NodeType.OUTPUT, label=finding_token, position=0)],
                edges=[("nonexistent", out, 1.0)],
            )

    with pytest.raises(KeyError, match="unknown source node"):
        build_attribution_graph(prompt="p", finding_token="effusion", backend=_BadBackend())


def test_build_then_prune_round_trips() -> None:
    g = build_attribution_graph(
        prompt="<image> effusion?",
        finding_token="effusion",
        backend=_MockBackend(with_image=True),
        image=object(),
    )
    pruned = prune(g, threshold=0.8)
    # pruning keeps the logit and shrinks the graph; metadata survives
    assert pruned.nodes_by_type(NodeType.OUTPUT)
    assert len(pruned.nodes) <= len(g.nodes)
    assert pruned.metadata["finding_token"] == "effusion"


def test_prune_drops_lowest_influence_node() -> None:
    g = _two_path_graph()
    pruned = prune(g, threshold=0.8)
    # the error node carries the least influence on the logit and is dropped first
    assert "err" not in pruned.nodes
    # the logit and the finding feature on the surviving paths are always retained
    assert "out" in pruned.nodes
    assert "feat_finding" in pruned.nodes
    assert len(pruned.nodes) < len(g.nodes)


def test_prune_keeps_only_edges_between_kept_nodes() -> None:
    pruned = prune(_two_path_graph(), threshold=0.8)
    for e in pruned.edges:
        assert e.source in pruned.nodes
        assert e.target in pruned.nodes


def test_prune_does_not_mutate_input() -> None:
    g = _two_path_graph()
    before_nodes, before_edges = len(g.nodes), len(g.edges)
    prune(g, threshold=0.5)
    assert len(g.nodes) == before_nodes
    assert len(g.edges) == before_edges
    assert "prune" not in g.metadata


def test_prune_records_metadata() -> None:
    pruned = prune(_two_path_graph(), threshold=0.8)
    meta = pruned.metadata["prune"]
    assert meta["n_nodes_before"] == 7
    assert meta["n_nodes_after"] == len(pruned.nodes)
    assert meta["n_edges_after"] == len(pruned.edges)
    assert 0.0 <= meta["retained_influence"] <= 1.0
    # original graph-level metadata is preserved
    assert pruned.metadata["finding_token"] == "effusion"


def test_prune_threshold_one_keeps_all_reachable_nodes() -> None:
    g = _two_path_graph()
    pruned = prune(g, threshold=1.0)
    # every node in this fixture has a path to the logit, so all survive
    assert set(pruned.nodes) == set(g.nodes)
    assert pruned.metadata["prune"]["retained_influence"] == pytest.approx(1.0)


def test_prune_threshold_zero_keeps_only_outputs() -> None:
    pruned = prune(_two_path_graph(), threshold=0.0)
    assert set(pruned.nodes) == {"out"}
    assert pruned.edges == []


def test_prune_empty_graph_is_safe() -> None:
    pruned = prune(AttributionGraph(), threshold=0.8)
    assert pruned.nodes == {}
    assert pruned.metadata["prune"]["n_nodes_after"] == 0


def test_prune_unreachable_node_is_dropped() -> None:
    # a feature with no path to any output carries zero influence -> always pruned
    g = _two_path_graph()
    g.add_node(Node("orphan", NodeType.FEATURE, label="disconnected", position=9))
    pruned = prune(g, threshold=1.0)
    assert "orphan" not in pruned.nodes
