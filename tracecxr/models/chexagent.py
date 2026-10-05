"""CheXagent-2 (3B) adapter — the size-matched cross-model convergence probe.

CheXagent-2-3b (~2.78 B params, Phi-based) is the closest-in-scale independently trained CXR
VLM to MedGemma 1.5 4B; the 8 B CheXagent would confound scale with architecture, so it is
deliberately not used. Unlike the Gemma / Qwen adapters, CheXagent ships **custom modeling
code** (``trust_remote_code=True``), loads as an ``AutoModelForCausalLM`` with a single
tokenizer (no separate processor), and takes its image as a **filesystem path** via
``tokenizer.from_list_format`` inside a ShareGPT-style conversation. We therefore override
loading, prompt construction, reasoning, and probing while reusing the shared reason-then-probe
flow (``HFVLMAdapter.generate``) and the yes/no logit reader from :mod:`._base`.

Reference usage (model card):
    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).to(torch.bfloat16)
    query = tokenizer.from_list_format([{'image': path}, {'text': prompt}])
    conv = [{"from": "system", "value": ...}, {"from": "human", "value": query}]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    out = model.generate(input_ids, do_sample=False, num_beams=1, max_new_tokens=512)[0]

VERIFY on first GPU run (this adapter could not be executed offline — no CheXagent weights /
GPU here):
  1. the assistant turn role in the multi-turn probe conv ("gpt" vs "assistant").
  2. that a probe forward pass ``model(input_ids).logits`` sees the image loaded from the path
     embedded via ``from_list_format`` (Qwen-VL-style internal image loading from the <img> tag).
  3. yes/no first-token ids resolve on CheXagent's tokenizer (uses the shared resolver).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tracecxr.core.model import register_model
from tracecxr.core.types import Finding

from ._base import (
    HFVLMAdapter,
    answer_readouts,
    resolve_yes_no_token_ids,
    yes_no_question,
    yes_probability,
)

SYSTEM_PROMPT = "You are an expert radiology assistant."


class CheXagentAdapter(HFVLMAdapter):
    """Reason-then-probe adapter for CheXagent-2-3b (custom modeling code)."""

    name = "chexagent"
    config_key = "chexagent"

    def __init__(
        self, *, model_id: str | None = None, device: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(model_id=model_id, device=device, max_new_tokens=max_new_tokens)
        self._tmp = Path(tempfile.mkdtemp(prefix="chexagent_img_"))

    # -- loading: AutoModelForCausalLM + single tokenizer, trust_remote_code --------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        from tracecxr.core.config import cache_root  # noqa: PLC0415

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = str(cache_root() / "models")
        tok = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, cache_dir=cache_dir
        )
        if self.device == "cuda":
            model = model.to(torch.bfloat16)
        self._model = model.to(self.device).eval()
        self._processor = tok  # single tokenizer stands in for the processor
        self._yes_ids, self._no_ids = resolve_yes_no_token_ids(tok)

    # -- image as a filesystem path (from_list_format wants a path) ----------------------

    def _image_path(self, image: Any | None) -> str | None:
        if image is None:
            return None
        from PIL import Image  # noqa: PLC0415

        pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        path = self._tmp / "current.png"
        pil.convert("RGB").save(path)
        return str(path)

    # ``_user_content`` returns an opaque payload so the base generate()/flow is reused.
    def _user_content(self, instruction: str, image: Any | None) -> dict[str, Any]:
        return {"instruction": instruction, "image_path": self._image_path(image)}

    def _query(self, payload: dict[str, Any]) -> str:
        parts: list[dict[str, str]] = []
        if payload["image_path"]:
            parts.append({"image": payload["image_path"]})
        parts.append({"text": payload["instruction"]})
        return self._processor.from_list_format(parts)

    def _encode(self, conv: list[dict[str, str]]) -> Any:
        enc = self._processor.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        return input_ids.to(self._model.device)

    def _reason(self, user_content: dict[str, Any]) -> str:
        import torch  # noqa: PLC0415

        conv = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": self._query(user_content)},
        ]
        input_ids = self._encode(conv)
        with torch.inference_mode():
            out = self._model.generate(
                input_ids, do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens, use_cache=True,
            )[0]
        return self._processor.decode(
            out[input_ids.size(1):], skip_special_tokens=True
        ).strip()

    def _probe(self, user_content: dict[str, Any], reasoning: str, finding: Finding) -> float:
        import torch  # noqa: PLC0415

        conv: list[dict[str, str]] = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": self._query(user_content)},
        ]
        if reasoning:
            conv.append({"from": "gpt", "value": reasoning})  # VERIFY role name on CheXagent-2
        conv.append({"from": "human", "value": yes_no_question(finding)})
        input_ids = self._encode(conv)
        with torch.inference_mode():
            logits_row = self._model(input_ids).logits[0, -1, :]
        return yes_probability(logits_row, self._yes_ids or [], self._no_ids or [])

    def _probe_single(self, user_content: dict[str, Any]) -> float:
        return self._probe_single_row(user_content)[1]

    def _probe_single_row(self, user_content: dict[str, Any]) -> tuple[float, float, float]:
        import torch  # noqa: PLC0415

        conv = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": self._query(user_content)},
        ]
        input_ids = self._encode(conv)
        with torch.inference_mode():
            logits_row = self._model(input_ids).logits[0, -1, :]
        return answer_readouts(logits_row, self._yes_ids or [], self._no_ids or [])


register_model("chexagent", CheXagentAdapter)
