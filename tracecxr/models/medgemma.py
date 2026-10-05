"""MedGemma real model adapter (primary subject).

MedGemma 1.5 4B-IT (``google/medgemma-1.5-4b-it``, resolved from ``core.config.MODELS``) is
the variant CheXthought evaluated and the project's primary subject — it exhibits
report-prior hallucination most strongly and its Gemma-3 decoder matches the released VLM
transcoder tooling. Inference is reason-then-probe; see :mod:`tracecxr.models._base` and
``docs/methodology.md``. Construction is offline; weights lazy-load on first use.
"""

from __future__ import annotations

from tracecxr.core.model import register_model

from ._base import HFVLMAdapter


class MedGemmaAdapter(HFVLMAdapter):
    """Reason-then-probe adapter for MedGemma."""

    name = "medgemma"
    config_key = "medgemma"

    def __init__(self, *, model_id: str | None = None, device: str | None = None,
                 max_new_tokens: int = 768) -> None:
        super().__init__(model_id=model_id, device=device, max_new_tokens=max_new_tokens)


register_model("medgemma", MedGemmaAdapter)
