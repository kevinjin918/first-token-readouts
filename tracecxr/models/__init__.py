"""Real VLM adapters for TraceCXR (Unit 1).

Importing this package registers the real model adapters under their names via
``register_model`` so :func:`tracecxr.core.available_models` lists them. Discovery imports
each submodule (see :func:`tracecxr.core.load_builtin_models`); importing the names here
makes ``from tracecxr.models import MedGemmaAdapter`` work and guarantees registration even
if a caller imports the package directly.

Construction is offline and weights are lazy-loaded; nothing here downloads a model.
"""

from __future__ import annotations

from .chexagent import CheXagentAdapter
from .medgemma import MedGemmaAdapter
from .qwen3vl import Qwen3VLAdapter

__all__ = ["CheXagentAdapter", "MedGemmaAdapter", "Qwen3VLAdapter"]
