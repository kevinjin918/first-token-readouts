"""The model-adapter contract and a registry for named adapters.

Every VLM the benchmark evaluates — mock or real — implements :class:`ModelAdapter`.
Adapters are looked up by name so the CLI and runner stay decoupled from concrete classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from tracecxr.core.types import CXRRecord, ModelOutput


@runtime_checkable
class ModelAdapter(Protocol):
    """A chest-X-ray vision-language model under evaluation.

    Implementations must support running *with* and *without* the image so the benchmark
    can isolate the report-prior signal (the ``with_image=False`` path is the no-image
    hallucination probe).
    """

    name: str

    def generate(
        self,
        record: CXRRecord,
        *,
        with_image: bool = True,
        prompt: str | None = None,
    ) -> ModelOutput:
        """Produce a finding report for ``record``.

        Args:
            record: the study to read.
            with_image: when False, the image is withheld (no-image probe).
            prompt: optional override prompt; adapters supply a sensible default.
        """
        ...


# --- Registry -------------------------------------------------------------------------

ModelFactory = Callable[..., ModelAdapter]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register a model factory under ``name`` (idempotent overwrite warns via ValueError)."""
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        raise ValueError(f"model {name!r} already registered with a different factory")
    _REGISTRY[name] = factory


def get_model(name: str, **kwargs: object) -> ModelAdapter:
    """Instantiate a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_models() -> list[str]:
    return sorted(_REGISTRY)
