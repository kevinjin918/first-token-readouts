"""Stage 4 causal validation by intervention.

Two layers live here. :mod:`tracecxr.intervention.patching` is the abstract constrained-
patching contract (:class:`PatchSpec` over the generic :class:`~tracecxr.core.ModelAdapter`),
per the manifesto's "Stage 4: causal validation by intervention". :mod:`tracecxr.intervention.clamp`
is the concrete engine that edits individual cross-layer-transcoder features inside MedGemma
(:class:`FeatureEdit`, :func:`apply_feature_edits`, :class:`ClampedMedGemma`) — the workhorse
for the bidirectional dial and the clamp-vs-no-clamp occlusion verification.
"""

from __future__ import annotations

from tracecxr.intervention.clamp import (
    ClampedMedGemma,
    FeatureEdit,
    apply_feature_edits,
)
from tracecxr.intervention.patching import (
    PatchSpec,
    apply_patch,
    constrained_patch,
    measure_effect,
)

__all__ = [
    "ClampedMedGemma",
    "FeatureEdit",
    "PatchSpec",
    "apply_feature_edits",
    "apply_patch",
    "constrained_patch",
    "measure_effect",
]
