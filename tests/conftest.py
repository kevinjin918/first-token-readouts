"""Shared pytest configuration.

Tests tagged ``requires_data`` (need a real dataset on disk) or ``requires_gpu`` (need a
GPU / real model weights) are skipped by default so the mock-first suite runs anywhere.
Opt in by setting ``TRACECXR_RUN_DATA=1`` / ``TRACECXR_RUN_GPU=1`` in the environment.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_data = os.environ.get("TRACECXR_RUN_DATA") == "1"
    run_gpu = os.environ.get("TRACECXR_RUN_GPU") == "1"
    skip_data = pytest.mark.skip(reason="needs a real dataset; set TRACECXR_RUN_DATA=1")
    skip_gpu = pytest.mark.skip(reason="needs a GPU / real weights; set TRACECXR_RUN_GPU=1")
    for item in items:
        if "requires_data" in item.keywords and not run_data:
            item.add_marker(skip_data)
        if "requires_gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
