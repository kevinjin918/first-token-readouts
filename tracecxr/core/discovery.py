"""Import-time discovery of self-registering components.

Metric / model / dataset modules register themselves on import (via the registries in
``tracecxr.core``). These helpers import every submodule of the relevant package so the
registries are populated, without any single unit having to edit a shared ``__init__``.
Packages that do not exist yet are tolerated, so units can land in any order.
"""

from __future__ import annotations

import importlib
import pkgutil


def _load_package(qualified_name: str) -> list[str]:
    """Import every submodule of ``qualified_name``; return the names imported.

    Returns an empty list if the package is not present (e.g. a unit hasn't landed yet).
    """
    try:
        pkg = importlib.import_module(qualified_name)
    except ModuleNotFoundError:
        return []
    if not hasattr(pkg, "__path__"):
        return [qualified_name]
    loaded: list[str] = []
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{qualified_name}.{mod.name}")
        loaded.append(mod.name)
    return loaded


def load_builtin_metrics() -> list[str]:
    """Import all modules under ``tracecxr.benchmark.metrics`` to register their metrics."""
    return _load_package("tracecxr.benchmark.metrics")


def load_builtin_models() -> list[str]:
    """Import all modules under ``tracecxr.models`` to register their adapters."""
    return _load_package("tracecxr.models")


def load_builtin_datasets() -> list[str]:
    """Import all modules under ``tracecxr.data`` to register their dataset loaders."""
    return _load_package("tracecxr.data")
