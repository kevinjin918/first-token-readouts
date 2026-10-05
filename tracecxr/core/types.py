"""Shared data contracts for TraceCXR.

Every unit of the codebase codes against the types defined here. Do not redefine these
elsewhere — import them from ``tracecxr.core``.

The finding taxonomy follows the 14 CheXpert observations. The project focuses on two
findings with high inter-reader agreement (per the manifesto): pleural effusion and
pneumothorax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

import numpy as np


class Finding(StrEnum):
    """The 14 CheXpert observations.

    ``EFFUSION`` and ``PNEUMOTHORAX`` are the project's focus pair.
    """

    NO_FINDING = "No Finding"
    ENLARGED_CARDIOMEDIASTINUM = "Enlarged Cardiomediastinum"
    CARDIOMEGALY = "Cardiomegaly"
    LUNG_OPACITY = "Lung Opacity"
    LUNG_LESION = "Lung Lesion"
    EDEMA = "Edema"
    CONSOLIDATION = "Consolidation"
    PNEUMONIA = "Pneumonia"
    ATELECTASIS = "Atelectasis"
    PNEUMOTHORAX = "Pneumothorax"
    EFFUSION = "Pleural Effusion"
    PLEURAL_OTHER = "Pleural Other"
    FRACTURE = "Fracture"
    SUPPORT_DEVICES = "Support Devices"


#: The two findings the project validates end-to-end (clean radiologist ground truth).
FOCUS_FINDINGS: tuple[Finding, Finding] = (Finding.EFFUSION, Finding.PNEUMOTHORAX)


class Label(int, Enum):
    """CheXpert-style label for a finding within a record."""

    NEGATIVE = 0
    POSITIVE = 1
    UNCERTAIN = -1
    BLANK = -2  # not mentioned


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (origin = top-left).

    Stored as (x, y, w, h) to match the NIH ChestX-ray14 ``BBox_List_2017.csv`` convention.
    """

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w < 0 or self.h < 0:
            raise ValueError(f"BBox width/height must be non-negative, got w={self.w}, h={self.h}")

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with another box. Returns 0.0 when both areas are zero."""
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class CXRRecord:
    """A single chest X-ray study and its annotations.

    ``image`` is an HxW (grayscale) or HxWxC ndarray. ``findings`` carries the reference
    label per finding; ``bboxes`` carries radiologist-drawn regions per finding (may be
    empty for report-only datasets such as CheXpert Plus / MIMIC).
    """

    id: str
    image: np.ndarray | None
    report: str | None = None
    findings: dict[Finding, Label] = field(default_factory=dict)
    bboxes: dict[Finding, list[BBox]] = field(default_factory=dict)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def label(self, finding: Finding) -> Label:
        """Reference label for ``finding``; ``Label.BLANK`` if not annotated."""
        return self.findings.get(finding, Label.BLANK)


@dataclass
class ModelOutput:
    """A VLM's response to one prompt.

    ``finding_predictions`` maps each finding the model commented on to a confidence in
    [0, 1]. ``abstained`` is True when the model declined to call any finding.
    """

    text: str
    finding_predictions: dict[Finding, float] = field(default_factory=dict)
    abstained: bool = False
    raw: Any = None

    def asserts(self, finding: Finding, threshold: float = 0.5) -> bool:
        """Whether the model positively asserts ``finding`` at or above ``threshold``."""
        return self.finding_predictions.get(finding, 0.0) >= threshold

    def asserted_findings(self, threshold: float = 0.5) -> set[Finding]:
        return {f for f, p in self.finding_predictions.items() if p >= threshold}


@dataclass
class MetricResult:
    """The outcome of evaluating one metric over one (model, dataset) pair."""

    key: str
    value: float
    per_finding: dict[Finding, float] = field(default_factory=dict)
    n: int = 0
    details: dict[str, Any] = field(default_factory=dict)
