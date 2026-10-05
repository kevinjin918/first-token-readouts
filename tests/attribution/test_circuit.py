"""Tests for reading a circuit out of an attribution graph (manifesto §Stage 2)."""

from __future__ import annotations

import pytest
from tracecxr.attribution import (
    AttributionGraph,
    Node,
    NodeType,
    features_by_attribution,
    report_prior_features,
)


def _graph() -> AttributionGraph:
    """Two text-position features and one image-position feature driving the logit."""
    g = AttributionGraph()
    g.add_node(Node("out", NodeType.OUTPUT, label="yes", activation=1.0, position=9))
    g.add_node(Node("img_in", NodeType.INPUT_IMAGE, label="patch", position=2))
    # report-prior (text) features
    g.add_node(Node("f_prior_hi", NodeType.FEATURE, label="prior-hi", position=4,
                    metadata={"layer": 10, "feature": 3}))
    g.add_node(Node("f_prior_lo", NodeType.FEATURE, label="prior-lo", position=5,
                    metadata={"layer": 11, "feature": 8}))
    # image-evidence (image-position) feature
    g.add_node(Node("f_visual", NodeType.FEATURE, label="visual", position=2,
                    metadata={"layer": 8, "feature": 1}))
    g.add_edge("f_prior_hi", "out", 1.4)
    g.add_edge("f_prior_lo", "out", 0.3)
    g.add_edge("f_visual", "out", 0.9)
    return g


def test_report_prior_picks_text_features_ranked() -> None:
    rows = report_prior_features(_graph(), top_k=8)
    # only the two text-position features, highest attribution first
    assert [(r.layer, r.feature) for r in rows] == [(10, 3), (11, 8)]
    assert rows[0].attribution == pytest.approx(1.4)


def test_report_prior_excludes_image_features() -> None:
    rows = report_prior_features(_graph())
    assert all(r.node_id != "f_visual" for r in rows)


def test_kind_image_selects_only_image_features() -> None:
    rows = features_by_attribution(_graph(), kind="image")
    assert [(r.layer, r.feature) for r in rows] == [(8, 1)]


def test_kind_all_includes_everything_ranked() -> None:
    rows = features_by_attribution(_graph(), kind="all")
    assert [r.node_id for r in rows] == ["f_prior_hi", "f_visual", "f_prior_lo"]


def test_top_k_truncates() -> None:
    assert len(features_by_attribution(_graph(), kind="all", top_k=2)) == 2


def test_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        features_by_attribution(_graph(), kind="bogus")


def test_no_features_returns_empty() -> None:
    g = AttributionGraph()
    g.add_node(Node("out", NodeType.OUTPUT, label="yes", position=0))
    assert report_prior_features(g) == []
