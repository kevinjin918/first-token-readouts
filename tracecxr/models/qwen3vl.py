"""Qwen3-VL-8B-Thinking real model adapter (secondary subject).

Qwen3-VL-8B-Thinking (``Qwen/Qwen3-VL-8B-Thinking``, resolved from ``core.config.MODELS``)
is the secondary model, kept for a clean line to CheXthought's published numbers. As a
"Thinking" variant it emits a long chain-of-thought during the reasoning pass — which is
exactly the thinking-vs-base condition CheXthought studies, and which the constrained
yes/no probe reads the answer from (see :mod:`tracecxr.models._base`). The larger default
generation budget gives that reasoning room to complete.
"""

from __future__ import annotations

from tracecxr.core.model import register_model

from ._base import HFVLMAdapter


class Qwen3VLAdapter(HFVLMAdapter):
    """Reason-then-probe adapter for Qwen3-VL-8B-Thinking."""

    name = "qwen3vl"
    config_key = "qwen3vl"

    def __init__(self, *, model_id: str | None = None, device: str | None = None,
                 max_new_tokens: int = 4096) -> None:
        super().__init__(model_id=model_id, device=device, max_new_tokens=max_new_tokens)


register_model("qwen3vl", Qwen3VLAdapter)
