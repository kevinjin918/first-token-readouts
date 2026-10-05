"""Stage 2 attribution-graph data structures and (stubbed) heavy computations.

This module implements the *interface* for the attribution graphs described in
``TraceCXR_manifesto.md`` §"Stage 2: attribution graphs". The data structures
(:class:`Node`, :class:`Edge`, :class:`AttributionGraph`) are real and usable: you can
build a graph by hand, query it by node type, and serialize it. :func:`prune` is fully
implemented (pure NumPy — no model needed): it reduces a raw graph to its high-influence
subgraph via the Neumann-series influence scoring from the circuit-tracing appendix. The
heavy computation that *produces* a graph from a local replacement model
(:func:`build_attribution_graph`) is the only remaining stub (it needs MedGemma + the
trained CLT on a GPU) and raises :class:`NotImplementedError` until that path lands.

Stage 2 recap (manifesto §Stage 2):

- An attribution graph attributes a finding token (e.g. the "effusion" token) back through
  the local replacement model built in Stage 1.
- Node types are: **output** nodes (candidate tokens covering 95% of probability mass,
  capped at 10), **feature** nodes (active cross-layer-transcoder features at each
  position), **input** nodes (token and image-patch embeddings), and **error** nodes
  (the unexplained residual the transcoder failed to reconstruct).
- Edge weights are linear attributions in the local replacement model:
  ``edge weight = source activation × virtual weight``, computed efficiently with a
  stop-gradient backward Jacobian. Each feature's preactivation equals the sum of its
  incoming edges.
- The raw graph is enormous (edges in the millions even for short prompts), so it is
  pruned (see :func:`prune`).

The structure worth surfacing is two competing paths into the finding logit: an
image-evidence path (image patches → low-level visual features → abstract finding
features) and a report-prior path (prompt / assistant-context features driving the
finding regardless of pixels). On a hallucinated case the image path should be silent
and the report-prior path should win.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class NodeType(StrEnum):
    """The kinds of node in a Stage 2 attribution graph (manifesto §Stage 2).

    Attributes:
        OUTPUT: A candidate output token (the tokens covering ~95% of probability mass,
            capped at 10). These are the logit nodes attribution flows *into*.
        FEATURE: An active cross-layer-transcoder (CLT) feature at a given token /
            image-patch position and layer.
        INPUT_TOKEN: A prompt or assistant-context token embedding (the report-prior
            entry point).
        INPUT_IMAGE: An image-patch embedding (the image-evidence entry point).
        ERROR: The unexplained residual the transcoder failed to reconstruct at a
            position/layer ("dark matter"); large error mass signals off-distribution
            input.
    """

    OUTPUT = "output"
    FEATURE = "feature"
    INPUT_TOKEN = "input_token"
    INPUT_IMAGE = "input_image"
    ERROR = "error"


@dataclass
class Node:
    """A node in an attribution graph (manifesto §Stage 2).

    Args:
        id: Unique identifier within a graph.
        node_type: Which of the Stage 2 node families this is.
        label: Human-readable description (e.g. an auto-interp / supernode label, or the
            literal token for output/input nodes).
        activation: The node's activation in the local replacement model. For a feature
            node this is its preactivation (the sum of incoming edges); for an output
            node, the logit / probability mass; ``None`` if not yet computed.
        position: Token / image-patch position the node lives at (sequence index), or
            ``None`` for position-agnostic nodes.
        metadata: Free-form extras (layer index, supernode id, top-activating examples,
            etc.).
    """

    id: str
    node_type: NodeType
    label: str = ""
    activation: float | None = None
    position: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-friendly dict."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "label": self.label,
            "activation": self.activation,
            "position": self.position,
            "metadata": dict(self.metadata),
        }


@dataclass
class Edge:
    """A directed, weighted attribution edge (manifesto §Stage 2).

    The weight is the linear attribution in the local replacement model,
    ``source activation × virtual weight``. It is signed: a negative weight is a
    *suppressing* contribution (the load-bearing sign for the report-prior phenotype,
    where an abstain feature is suppressed).

    Args:
        source: ``id`` of the upstream node (the contributor).
        target: ``id`` of the downstream node (whose preactivation it feeds).
        weight: Signed attribution. ``target``'s preactivation is the sum of the
            weights of its incoming edges.
    """

    source: str
    target: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-friendly dict."""
        return {"source": self.source, "target": self.target, "weight": self.weight}


@dataclass
class AttributionGraph:
    """A Stage 2 attribution graph: nodes keyed by id, plus a list of edges.

    This is a plain container — the heavy lifting that *fills* it lives in
    :func:`build_attribution_graph` and :func:`prune`. Build small graphs by hand with
    :meth:`add_node` / :meth:`add_edge`.

    Attributes:
        nodes: Mapping from node id to :class:`Node`.
        edges: List of :class:`Edge`.
        metadata: Free-form graph-level extras (prompt, finding token, prune settings,
            completeness / replacement scores, etc.).
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> Node:
        """Add ``node`` to the graph, returning it.

        Raises:
            ValueError: If a node with the same ``id`` already exists.
        """
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id!r}")
        self.nodes[node.id] = node
        return node

    def add_edge(self, source: str, target: str, weight: float) -> Edge:
        """Add a weighted edge from ``source`` to ``target``, returning it.

        Both endpoints must already be present as nodes.

        Raises:
            KeyError: If either endpoint id is not a node in the graph.
        """
        if source not in self.nodes:
            raise KeyError(f"unknown source node: {source!r}")
        if target not in self.nodes:
            raise KeyError(f"unknown target node: {target!r}")
        edge = Edge(source=source, target=target, weight=weight)
        self.edges.append(edge)
        return edge

    def nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Return the nodes of a given :class:`NodeType`, in insertion order."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole graph to a plain JSON-friendly dict.

        The result round-trips through :meth:`from_dict`.
        """
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionGraph:
        """Reconstruct a graph from :meth:`to_dict` output."""
        graph = cls(metadata=dict(data.get("metadata", {})))
        for nd in data.get("nodes", []):
            graph.add_node(
                Node(
                    id=nd["id"],
                    node_type=NodeType(nd["node_type"]),
                    label=nd.get("label", ""),
                    activation=nd.get("activation"),
                    position=nd.get("position"),
                    metadata=dict(nd.get("metadata", {})),
                )
            )
        for ed in data.get("edges", []):
            graph.add_edge(ed["source"], ed["target"], ed["weight"])
        return graph


# --- canonical node-id scheme (shared by every backend and the assembler) ------------
# A backend and the assembler must agree on node ids so edges resolve. These helpers are
# the single source of truth; build a feature node's id with `feature_id(...)`, etc.


def feature_id(layer: int, position: int, feature: int) -> str:
    """Canonical id for a CLT feature node at ``(layer, position, feature)``."""
    return f"feat:L{layer}:p{position}:f{feature}"


def output_id(position: int, token: str) -> str:
    """Canonical id for an output (logit) node — a candidate ``token`` at ``position``."""
    return f"out:p{position}:{token}"


def input_token_id(position: int) -> str:
    """Canonical id for a prompt/context token-embedding input node at ``position``."""
    return f"in_tok:p{position}"


def input_image_id(position: int) -> str:
    """Canonical id for an image-patch-embedding input node at ``position``."""
    return f"in_img:p{position}"


def error_id(layer: int, position: int) -> str:
    """Canonical id for a reconstruction-error ("dark matter") node at ``(layer, position)``."""
    return f"err:L{layer}:p{position}"


@dataclass
class RawAttribution:
    """The raw output of an :class:`AttributionBackend` for one case (manifesto §Stage 2).

    A backend produces the fully-typed :class:`Node` list (with canonical ids from the
    ``*_id`` helpers, activations, positions, and per-node ``metadata`` such as ``layer``)
    and the weighted edges between them as ``(source_id, target_id, weight)`` triples.
    :func:`build_attribution_graph` validates and assembles these into an
    :class:`AttributionGraph`; ``metadata`` is merged onto the graph (e.g. replacement /
    completeness scores the backend computed).
    """

    nodes: list[Node]
    edges: list[tuple[str, str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AttributionBackend(Protocol):
    """Numerical engine that produces a :class:`RawAttribution` for one case.

    This is the seam between the pure graph assembly (this module, CI-tested) and the
    heavy, model-specific computation. The production backend (MedGemma with its MLPs
    replaced by the trained cross-layer transcoder, attention + layernorm frozen, per-
    position error terms — the Stage 1 local replacement model) computes node activations
    and the linear edge weights ``source activation × virtual weight`` via a stop-gradient
    backward Jacobian; it lives in its own torch-only module and runs only on a GPU. Tests
    inject a deterministic mock that satisfies this same protocol.
    """

    def trace(
        self,
        *,
        prompt: str,
        finding_token: str,
        image: Any | None = None,
        n_logits: int = 10,
        logit_mass: float = 0.95,
    ) -> RawAttribution:
        """Trace the ``finding_token`` logit back through the replacement model."""
        ...


def build_attribution_graph(
    *,
    prompt: str,
    finding_token: str,
    backend: AttributionBackend | None = None,
    image: Any | None = None,
    n_logits: int = 10,
    logit_mass: float = 0.95,
) -> AttributionGraph:
    """Build the raw attribution graph for one hallucination case (manifesto §Stage 2).

    Attributes back from the ``finding_token`` logit through the Stage 1 local
    replacement model (CLT-substituted MLPs, frozen attention, per-position error terms),
    assembling the :class:`RawAttribution` a ``backend`` produces into a validated
    :class:`AttributionGraph`. The resulting graph contains:

    - **output** nodes: the candidate tokens covering ``logit_mass`` (~0.95) of
      probability mass, capped at ``n_logits`` (~10);
    - **feature** nodes: the active CLT features at every token / image-patch position;
    - **input** nodes: token (:attr:`NodeType.INPUT_TOKEN`) and image-patch
      (:attr:`NodeType.INPUT_IMAGE`) embeddings;
    - **error** nodes: the unexplained reconstruction residual per position/layer.

    Edge weights are the linear attributions ``source activation × virtual weight``,
    obtained with a stop-gradient backward Jacobian so each feature's preactivation is
    exactly the sum of its incoming edges. The raw graph has up to millions of edges and
    is meant to be pruned with :func:`prune` before inspection.

    The numerical work lives behind ``backend`` (:class:`AttributionBackend`): the
    production backend runs MedGemma + the trained CLT on a GPU; tests pass a deterministic
    mock. Assembly here is pure — it validates ids (every edge endpoint must be a declared
    node; duplicate ids are rejected) and records the case metadata.

    Args:
        prompt: The full prompt (including image placeholder) that elicited the finding.
        finding_token: The output token to attribute from (e.g. ``"effusion"``).
        backend: The engine that computes nodes + edge weights. Required — the GPU
            implementation (MedGemma + CLT) is a separate torch-only module.
        image: The CXR passed through to the backend (``None`` for the no-image case).
        n_logits: Cap on the number of output nodes.
        logit_mass: Probability mass the output nodes should cover.

    Returns:
        The (un-pruned) :class:`AttributionGraph`.

    Raises:
        NotImplementedError: If ``backend`` is ``None`` — there is no built-in numerical
            engine; supply one (the GPU backend lands as its own module).
        ValueError: If the backend emits a duplicate node id.
        KeyError: If an edge references an undeclared node id.
    """
    if backend is None:
        raise NotImplementedError(
            "build_attribution_graph needs an AttributionBackend; the GPU engine "
            "(MedGemma + the trained CLT) is a separate torch-only module. See "
            "TraceCXR_manifesto.md §Stage 2."
        )
    raw = backend.trace(
        prompt=prompt,
        finding_token=finding_token,
        image=image,
        n_logits=n_logits,
        logit_mass=logit_mass,
    )
    graph = AttributionGraph(
        metadata={
            "prompt": prompt,
            "finding_token": finding_token,
            "n_logits": n_logits,
            "logit_mass": logit_mass,
            "has_image": image is not None,
            **raw.metadata,
        }
    )
    for node in raw.nodes:
        graph.add_node(node)
    for source, target, weight in raw.edges:
        graph.add_edge(source, target, weight)
    return graph


def _influence_on_outputs(graph: AttributionGraph) -> dict[str, float]:
    """Total influence of every node on the output (logit) nodes (manifesto §Stage 2).

    Forms the normalized unsigned adjacency ``A`` where ``A[t, s]`` is the absolute
    weight of edge ``s → t`` divided by the sum of absolute incoming weights at ``t`` (so
    each node's incoming weights sum to 1), then accumulates the influence that flows from
    each node into the outputs via the Neumann series ``(I − A)⁻¹ = Σ_k A^k``. Concretely
    it solves ``w = p + Aᵀ w`` iteratively, where ``p`` weights each output node by its
    own activation (probability mass), uniform if no activations are set. The returned
    influence of node ``n`` is ``w[n] − p[n]`` (the indirect ``B = (I−A)⁻¹ − I`` term),
    clipped at 0. For the frozen-attention replacement model the graph is a DAG, so the
    series terminates exactly in ``depth`` iterations.
    """
    ids = list(graph.nodes.keys())
    n = len(ids)
    idx = {nid: i for i, nid in enumerate(ids)}
    influence = dict.fromkeys(ids, 0.0)
    if n == 0:
        return influence

    a = np.zeros((n, n), dtype=np.float64)  # a[t, s] = normalized weight of edge s -> t
    incoming_abs = np.zeros(n, dtype=np.float64)
    for e in graph.edges:
        if e.source in idx and e.target in idx and np.isfinite(e.weight):
            incoming_abs[idx[e.target]] += abs(e.weight)
    for e in graph.edges:
        if e.source in idx and e.target in idx and np.isfinite(e.weight):
            t = idx[e.target]
            if incoming_abs[t] > 0:
                a[t, idx[e.source]] += abs(e.weight) / incoming_abs[t]

    p = np.zeros(n, dtype=np.float64)
    outputs = [nid for nid in ids if graph.nodes[nid].node_type == NodeType.OUTPUT]
    if not outputs:
        return influence
    acts = [graph.nodes[o].activation for o in outputs]
    if all(act is not None and abs(act) > 0 for act in acts):
        for o, act in zip(outputs, acts, strict=True):
            p[idx[o]] = abs(float(act))
    else:  # no usable activations -> weight outputs uniformly
        for o in outputs:
            p[idx[o]] = 1.0

    at = a.T
    w = p.copy()
    for _ in range(n + 1):  # DAG: converges in <= depth steps; +1 guards the bound
        nxt = p + at @ w
        if np.allclose(nxt, w, rtol=1e-9, atol=1e-12):
            w = nxt
            break
        w = nxt

    infl = np.clip(w - p, 0.0, None)
    return {nid: float(infl[idx[nid]]) for nid in ids}


def prune(graph: AttributionGraph, threshold: float = 0.8) -> AttributionGraph:
    """Prune an attribution graph to its high-influence subgraph (manifesto §Stage 2).

    Follows the pruning appendix: form the normalized unsigned adjacency matrix ``A``
    (take absolute edge weights, normalize each node's incoming edges to sum to 1),
    accumulate the indirect-influence ``B = (I − A)⁻¹ − I`` (the Neumann series, which
    sums the strength of all paths between node pairs — see :func:`_influence_on_outputs`),
    and score each node by its total influence on the logit (output) nodes. Output nodes
    are always kept; the remaining nodes are kept in descending-influence order until the
    retained ``threshold`` fraction of total node influence is covered. Nodes with no path
    to any output (zero influence) are always dropped.

    The default (~0.8) reduces the node count substantially while retaining ~80% of
    explained behavior. Kept edges are those whose endpoints both survive. The retained
    influence fraction and before/after counts are recorded in the returned graph's
    :attr:`~AttributionGraph.metadata`.

    Args:
        graph: The raw graph from :func:`build_attribution_graph`.
        threshold: Fraction of total node influence to retain (~0.8 default); clamped to
            ``[0, 1]``.

    Returns:
        A new, pruned :class:`AttributionGraph` (the input is not mutated).
    """
    threshold = min(max(threshold, 0.0), 1.0)
    influence = _influence_on_outputs(graph)

    output_ids = {nid for nid, nd in graph.nodes.items() if nd.node_type == NodeType.OUTPUT}
    candidates = [
        (nid, influence[nid])
        for nid in graph.nodes
        if nid not in output_ids and influence[nid] > 0
    ]
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    total = sum(infl for _, infl in candidates)

    kept = set(output_ids)
    retained = 0.0
    if total > 0:
        target = threshold * total
        cumulative = 0.0
        for nid, infl in candidates:
            if cumulative >= target:
                break
            kept.add(nid)
            cumulative += infl
        retained = cumulative / total

    pruned = AttributionGraph(metadata=dict(graph.metadata))
    for nid, nd in graph.nodes.items():
        if nid in kept:
            pruned.add_node(replace(nd, metadata=dict(nd.metadata)))
    for e in graph.edges:
        if e.source in kept and e.target in kept:
            pruned.add_edge(e.source, e.target, e.weight)
    pruned.metadata["prune"] = {
        "threshold": threshold,
        "n_nodes_before": len(graph.nodes),
        "n_nodes_after": len(pruned.nodes),
        "n_edges_before": len(graph.edges),
        "n_edges_after": len(pruned.edges),
        "retained_influence": retained,
    }
    return pruned
