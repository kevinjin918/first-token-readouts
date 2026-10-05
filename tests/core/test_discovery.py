"""Discovery helpers tolerate packages that haven't landed yet and import real ones."""

from __future__ import annotations

from tracecxr.core import (
    available_models,
    load_builtin_datasets,
    load_builtin_metrics,
    load_builtin_models,
)


def test_load_builtin_metrics_tolerant():
    # The metrics package exists but may be empty until Units 7-11 land; must not raise.
    assert isinstance(load_builtin_metrics(), list)


def test_load_builtin_datasets_tolerant():
    assert isinstance(load_builtin_datasets(), list)


def test_load_builtin_models_tolerant():
    # The models package may not exist yet (Unit 1); must return a list, not raise.
    loaded = load_builtin_models()
    assert isinstance(loaded, list)
    # The mock model is always registered by core, independent of discovery.
    assert "mock" in available_models()
