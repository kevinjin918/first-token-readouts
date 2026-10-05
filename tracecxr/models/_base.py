"""Shared scaffolding for real HuggingFace VLM adapters (MedGemma, Qwen3-VL).

Finding extraction is **constrained yes/no probing**, not text parsing — see
``docs/methodology.md`` for the full justification. In short: the model reasons freely
(CheXthought's "step-by-step diagnostic reasoning" prompt, preserved so the thinking-vs-base
effect survives), then for each finding we read the next-token logits of a closed
"is there <finding>? yes/no" question and report ``P(yes) = softmax([logit_yes, logit_no])``.
There is no report text to parse, so the format-drift / truncation failures of keyword and
verdict parsing are gone. The free reasoning is still returned in ``ModelOutput.text`` for
the mechanistic stages.

Both concrete adapters share the same transformers classes and flow, so the whole pipeline
lives in :class:`HFVLMAdapter`; the subclasses only set a name, a config key, and a default
generation budget. torch / transformers are imported lazily inside ``_ensure_loaded`` /
``_reason`` / probing, never at module top, so importing this package is offline.
"""

from __future__ import annotations

from typing import Any

from tracecxr.core.config import MODELS, cache_root
from tracecxr.core.types import CXRRecord, Finding, ModelOutput

#: Findings probed per record — the CheXpert pathologies CheXthought's occlusion study covers
#: (excludes NO_FINDING / SUPPORT_DEVICES / PLEURAL_OTHER / ENLARGED_CARDIOMEDIASTINUM).
PROBE_FINDINGS: tuple[Finding, ...] = (
    Finding.EFFUSION,
    Finding.PNEUMOTHORAX,
    Finding.CARDIOMEGALY,
    Finding.ATELECTASIS,
    Finding.CONSOLIDATION,
    Finding.EDEMA,
    Finding.PNEUMONIA,
    Finding.LUNG_OPACITY,
    Finding.LUNG_LESION,
    Finding.FRACTURE,
)

#: Reasoning elicitation — CheXthought's prompt verbatim. No output-format demand: the
#: answer is read from a separate constrained probe, not from this free text.
REASONING_PROMPT: str = (
    "Interpret this chest X-ray and provide step-by-step diagnostic reasoning."
)


def build_radiology_prompt(prompt: str | None = None) -> str:
    """Return ``prompt`` if provided, else the default reasoning elicitation."""
    return prompt if prompt is not None else REASONING_PROMPT


def yes_no_question(finding: Finding) -> str:
    """The closed probe question for one finding."""
    return (
        f"Based on your analysis of this chest X-ray, is there {finding.value.lower()}? "
        "Answer with a single word: yes or no."
    )


def resolve_yes_no_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """First-token ids for yes/no answer variants, used to read the probe logits.

    Different tokenizers split "yes"/"no" with or without a leading space and casing; we
    take the first token of each surface form so the probe reads the actual answer token.
    """
    yes_words = ("yes", " yes", "Yes", " Yes", "YES")
    no_words = ("no", " no", "No", " No", "NO")

    def first_ids(words: tuple[str, ...]) -> list[int]:
        ids: set[int] = set()
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return sorted(ids)

    return first_ids(yes_words), first_ids(no_words)


def yes_probability(logits_row: Any, yes_ids: list[int], no_ids: list[int]) -> float:
    """P(yes) from a next-token logit vector, restricted to the yes/no answer tokens.

    Takes the max logit within each group (any yes-surface vs any no-surface) and softmaxes
    the pair. Pure tensor math — unit-tested with a fake logit row.
    """
    import torch  # noqa: PLC0415

    yes_logit = logits_row[yes_ids].max()
    no_logit = logits_row[no_ids].max()
    pair = torch.stack([yes_logit, no_logit]).float()
    return float(torch.softmax(pair, dim=0)[0])


def answer_readouts(logits_row: Any, yes_ids: list[int],
                    no_ids: list[int]) -> tuple[float, float, float]:
    """The three first-token numbers a closed-answer probe can report, from one logit row.

    Returns ``(raw, normalised, mass)``:

    - ``raw`` -- P(yes) under the full-vocabulary softmax, i.e. the probability the very next token
      is an affirmative surface form. This is what most intervention and evaluation code reports,
      and it mixes the yes/no preference with the propensity to answer in that format at all.
    - ``normalised`` -- P(yes | the next token is a yes or no form).
    - ``mass`` -- P(yes) + P(no), the share of next-token probability landing on an answer. On
      instruction-tuned models this is often well below 1 (0.26 on one of our film sets), which is
      exactly when ``raw`` and ``normalised`` come apart.

    None of the three is the model's answer. When mass is low, the answer is decided after an
    opening such as "Based on the image, ...", and ``normalised`` ignores that route entirely: in
    results/clt_scale/readout_generation_check.md a clamp left ``normalised`` at 0.997 on 20
    films, yet the greedy answer turned to no on 4 of them and the sampled yes-rate fell from 0.98
    to 0.73. Score the generated answer (``tracecxr.intervention.generation_check``) when an
    intervention or an evaluation depends on it, and report these alongside.
    """
    import torch  # noqa: PLC0415

    probs = torch.softmax(logits_row.float(), dim=-1)
    raw = float(probs[yes_ids].max())
    mass = raw + float(probs[no_ids].max())
    return raw, yes_probability(logits_row, yes_ids, no_ids), mass


class HFVLMAdapter:
    """Reason-then-probe :class:`~tracecxr.core.model.ModelAdapter` over a HF VLM.

    Subclasses set :attr:`name`, :attr:`config_key` (key into ``core.config.MODELS``) and a
    default ``max_new_tokens``. Construction is offline; weights load on first use.
    """

    name: str = "hf-vlm"
    config_key: str = ""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id or MODELS[self.config_key].locator
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    # -- Lazy weight loading -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        cache_dir = str(cache_root() / "models")
        self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=dtype, cache_dir=cache_dir
        ).to(self.device)
        self._model.eval()
        self._yes_ids, self._no_ids = resolve_yes_no_token_ids(self._processor.tokenizer)

    # -- Inference: reason once, then probe each finding -------------------------------

    def generate(
        self,
        record: CXRRecord,
        *,
        with_image: bool = True,
        prompt: str | None = None,
    ) -> ModelOutput:
        """Reason over ``record``, then read a yes/no probe per finding.

        ``finding_predictions[f]`` is ``P(present)`` from the constrained probe (conditioned
        on the model's reasoning). ``with_image=False`` withholds the image (the no-image
        probe). The free reasoning is returned in ``text`` for the mechanistic stages.
        """
        self._ensure_loaded()
        include_image = with_image and record.image is not None
        image = record.image if include_image else None
        instruction = build_radiology_prompt(prompt)

        user_content = self._user_content(instruction, image)
        reasoning = self._reason(user_content)
        preds = {f: self._probe(user_content, reasoning, f) for f in PROBE_FINDINGS}
        abstained = all(p < 0.5 for p in preds.values())
        return ModelOutput(
            text=reasoning,
            finding_predictions=preds,
            abstained=abstained,
            raw={"model_id": self.model_id, "with_image": include_image},
        )

    def _user_content(self, instruction: str, image: Any | None) -> list[dict[str, Any]]:
        from PIL import Image  # noqa: PLC0415

        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        if image is not None:
            pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
            content.insert(0, {"type": "image", "image": pil})
        return content

    def _reason(self, user_content: list[dict[str, Any]]) -> str:
        import torch  # noqa: PLC0415

        messages = [{"role": "user", "content": user_content}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = out[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _probe(self, user_content: list[dict[str, Any]], reasoning: str, finding: Finding) -> float:
        import torch  # noqa: PLC0415

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": reasoning}]},
            {"role": "user", "content": [{"type": "text", "text": yes_no_question(finding)}]},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            logits_row = self._model(**inputs).logits[0, -1, :]
        return yes_probability(logits_row, self._yes_ids or [], self._no_ids or [])

    # -- Single-shot probe (no free reasoning) -----------------------------------------

    def probe_finding(
        self, record: CXRRecord, finding: Finding, *,
        with_image: bool = True, prompt: str | None = None,
    ) -> float:
        """Single-shot ``P(finding present)``: one user turn (image + a yes/no question), no
        free reasoning. Matches the banked single-shot shortcut measurement; ~1 forward pass."""
        self._ensure_loaded()
        include_image = with_image and record.image is not None
        image = record.image if include_image else None
        question = prompt if prompt is not None else yes_no_question(finding)
        return self._probe_single(self._user_content(question, image))

    def _probe_single(self, user_content: list[dict[str, Any]]) -> float:
        return self._probe_single_row(user_content)[1]

    def probe_readouts(self, question: str, image: Any | None = None) -> tuple[float, float, float]:
        """Single-shot ``(raw, normalised, mass)`` for a yes/no question. See answer_readouts."""
        self._ensure_loaded()
        return self._probe_single_row(self._user_content(question, image))

    def _probe_single_row(self, user_content: list[dict[str, Any]]) -> tuple[float, float, float]:
        import torch  # noqa: PLC0415

        messages = [{"role": "user", "content": user_content}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            logits_row = self._model(**inputs).logits[0, -1, :]
        return answer_readouts(logits_row, self._yes_ids or [], self._no_ids or [])
