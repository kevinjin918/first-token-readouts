"""The metric contract and a registry for named faithfulness metrics.

A metric consumes a model and an iterable of records and returns a single
:class:`~tracecxr.core.types.MetricResult`. Metrics self-register via :func:`register_metric`
so the runner can discover them without importing each module directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from tracecxr.core.model import ModelAdapter
from tracecxr.core.types import CXRRecord, MetricResult


@runtime_checkable
class Metric(Protocol):
    """A faithfulness metric over a (model, dataset) pair."""

    key: str

    def compute(
        self,
        model: ModelAdapter,
        dataset: Iterable[CXRRecord],
        **opts: object,
    ) -> MetricResult:
        ...


# --- Registry -------------------------------------------------------------------------

_REGISTRY: dict[str, Metric] = {}


def register_metric(metric: Metric) -> Metric:
    """Register a metric instance keyed by its ``key``; returns the metric."""
    key = metric.key
    if key in _REGISTRY and _REGISTRY[key] is not metric:
        raise ValueError(f"metric {key!r} already registered")
    _REGISTRY[key] = metric
    return metric


def get_metric(key: str) -> Metric:
    if key not in _REGISTRY:
        raise KeyError(f"unknown metric {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def iter_metrics(keys: Iterable[str] | None = None) -> list[Metric]:
    """All registered metrics, or just those named in ``keys`` (in the given order)."""
    if keys is None:
        return [_REGISTRY[k] for k in sorted(_REGISTRY)]
    return [get_metric(k) for k in keys]


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)
