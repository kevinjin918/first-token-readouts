"""Constrained patching for Stage 4 causal validation.

An attribution graph is a hypothesis about the local-replacement model, which is only a
proxy for the real VLM. Per the manifesto's "Stage 4: causal validation by intervention",
the graph is validated by *perturbing the real model* and checking the effect matches the
graph, following the intervention method in the circuit-tracing paper.

The technique is **constrained patching**: multiply a target feature's activation by a
factor over a layer range, then run the forward pass *from the last layer of the range* so
that the only change to downstream MLP outputs is the intervention itself. The manifesto's
key guidance, baked into :class:`PatchSpec`:

* **Steer negatively, do not zero-ablate.** For cross-layer features the paper found that
  multiplying by ``-1.0`` (negative steering) is the right perturbation, rather than zeroing
  the activation. ``PatchSpec.factor`` defaults to ``-1.0`` for this reason.
* **Multiply over a layer range, run forward from its end.** The factor is applied across
  ``[layer_start, layer_end]`` and the forward pass resumes from ``layer_end`` so MLP
  outputs change only because of the patch.

These contracts are real; the heavy mechanics (hooking the model, running the constrained
forward pass, scoring the effect) are stubs until Stage 4 is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from tracecxr.core import ModelAdapter

_NOT_IMPLEMENTED = "Stage 4 not yet implemented; see TraceCXR_manifesto.md §Stage 4"


@dataclass(frozen=True, slots=True)
class PatchSpec:
    """A single constrained-patching intervention on one transcoder feature.

    Per the manifesto's "Stage 4: causal validation by intervention", a patch multiplies the
    target feature's activation by ``factor`` over the layer range ``[layer_start, layer_end]``
    and the constrained forward pass is then run *from the last layer of the range* so that
    the only change to downstream MLP outputs is this intervention.

    The manifesto's guidance for cross-layer features is to **steer negatively** (multiply by
    ``-1.0``) rather than zero-ablate; ``factor`` therefore defaults to ``-1.0``. A clean-case
    *activation* (the other half of the bidirectional gold standard) is expressed with a
    positive ``factor``.

    Attributes:
        feature_id: index of the target transcoder feature to perturb.
        factor: multiplier applied to the feature's activation. ``-1.0`` (the default) is the
            recommended negative-steering suppression; a positive value activates/amplifies
            the feature to manufacture a hallucination on a clean case.
        layer_start: first layer of the range over which ``factor`` is applied (inclusive).
        layer_end: last layer of the range (inclusive); the constrained forward pass resumes
            from here.
    """

    feature_id: int
    factor: float = -1.0
    layer_start: int = 0
    layer_end: int = 0


def apply_patch(
    model: ModelAdapter,
    spec: PatchSpec,
    *,
    record: object | None = None,
) -> object:
    """Run ``model`` under the constrained patch described by ``spec``.

    Per the manifesto's "Stage 4: causal validation by intervention", this multiplies the
    target feature's activation by ``spec.factor`` over ``[spec.layer_start, spec.layer_end]``
    and runs the forward pass *from the last layer of the range*, so the only change to MLP
    outputs is the intervention (negative steering, not zero-ablation, for cross-layer
    features).

    Args:
        model: the real VLM to perturb (the graph is validated against the real model, not the
            local-replacement proxy).
        spec: the constrained-patching intervention to apply.
        record: optional study/prompt context to run the patched forward pass on.

    Returns:
        The patched model output (type finalized when Stage 4 lands).

    Raises:
        NotImplementedError: always — this is a Stage 4 stub.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def constrained_patch(
    model: ModelAdapter,
    specs: list[PatchSpec],
    *,
    record: object | None = None,
) -> object:
    """Apply one or more constrained patches and return the perturbed forward pass.

    Per the manifesto's "Stage 4: causal validation by intervention", candidate features are
    prioritized by their **graph influence score**, which the paper shows predicts ablation
    effects (Spearman ~0.72 for feature-to-feature effects), so callers test high-influence
    features first. Each patch multiplies its feature over a layer range and the forward pass
    is run from the last layer of the range so the only change is the intervention.

    Args:
        model: the real VLM to perturb.
        specs: the ranked set of constrained-patching interventions to apply together.
        record: optional study/prompt context for the patched forward pass.

    Returns:
        The perturbed forward pass (type finalized when Stage 4 lands).

    Raises:
        NotImplementedError: always — this is a Stage 4 stub.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def measure_effect(
    baseline: object,
    patched: object,
    *,
    spec: PatchSpec | None = None,
) -> object:
    """Quantify how a constrained patch moved the model, against the graph's prediction.

    Per the manifesto's "Stage 4: causal validation by intervention", trust **near-downstream**
    and **final-output** effects most: the faithfulness check found perturbation effects match
    the real model well one layer downstream (~0.8 cosine) but degrade further down, so the
    model's actual output is the most reliable validation signal. The target is the
    **bidirectional gold standard** from the entity-hallucination study — suppress the override
    feature and the hallucination drops; activate it on a clean case and you manufacture a
    hallucination.

    Args:
        baseline: the unperturbed forward pass / output.
        patched: the perturbed forward pass / output from :func:`apply_patch` or
            :func:`constrained_patch`.
        spec: optional patch that produced ``patched``, for attributing the measured effect.

    Returns:
        The measured causal effect (type finalized when Stage 4 lands).

    Raises:
        NotImplementedError: always — this is a Stage 4 stub.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)
