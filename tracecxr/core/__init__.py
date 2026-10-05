"""TraceCXR shared substrate — the contracts every unit imports.

Import shared types, the model/metric/dataset protocols, their registries, and the mock VLM
from here. Do not redefine these elsewhere.
"""

from __future__ import annotations

from tracecxr.core.dataset import (
    Dataset,
    FixtureDataset,
    available_datasets,
    load_dataset,
    materialize_image,
    register_dataset,
)
from tracecxr.core.discovery import (
    load_builtin_datasets,
    load_builtin_metrics,
    load_builtin_models,
)
from tracecxr.core.metric import (
    Metric,
    available_metrics,
    get_metric,
    iter_metrics,
    register_metric,
)
from tracecxr.core.mock import MockVLM
from tracecxr.core.model import (
    ModelAdapter,
    available_models,
    get_model,
    register_model,
)
from tracecxr.core.types import (
    FOCUS_FINDINGS,
    BBox,
    CXRRecord,
    Finding,
    Label,
    MetricResult,
    ModelOutput,
)

__all__ = [
    "FOCUS_FINDINGS",
    "BBox",
    "CXRRecord",
    "Dataset",
    "FixtureDataset",
    "Finding",
    "Label",
    "Metric",
    "MetricResult",
    "MockVLM",
    "ModelAdapter",
    "ModelOutput",
    "available_datasets",
    "available_metrics",
    "available_models",
    "get_metric",
    "get_model",
    "iter_metrics",
    "load_builtin_datasets",
    "load_builtin_metrics",
    "load_builtin_models",
    "load_dataset",
    "materialize_image",
    "register_dataset",
    "register_metric",
    "register_model",
]
