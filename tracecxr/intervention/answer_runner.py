"""GPU side of the sampled-answer experiments: one loaded MedGemma (+ CLT), many arms.

Shared by ``scripts/readout_sampled_answers.py`` and ``scripts/clamp_contrastive_answers.py`` so the
readouts, the prefill gate, the feature captures and the sampling settings cannot drift between
them. Every method mirrors ``scripts/readout_generation_check.py`` (E2), which this generalises:
same readouts, same parse rule, same image fills, same feature capture, same clamp positions.

Pure helpers (``block_patch_ids``, ``block_slice``, ``fill``, ``patient``, ``draw_films``) are
importable without torch and unit-tested; :class:`AnswerRunner` needs a GPU and the CLT checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .generation_check import opening_word, parse_answer

GRID, BLOCK = 16, 2
NB = GRID // BLOCK
PROMPTS = {
    "cardiomegaly": "Does this chest X-ray show cardiomegaly? Answer yes or no.",
    "effusion": "Does this chest X-ray show a pleural effusion? Answer yes or no.",
    "pneumothorax": "Does this chest X-ray show a pneumothorax? Answer yes or no.",
}
LABELS = {"cardiomegaly": "Cardiomegaly", "effusion": "Effusion", "pneumothorax": "Pneumothorax"}


def patient(name: str) -> int:
    """NIH patient id from an image name, ``00000013_006.png`` -> 13."""
    return int(name.split("_")[0])


def block_patch_ids(block: int) -> list[int]:
    """Indices, in the row-major 16x16 image-token grid, of the 4 tokens under one 8x8 block."""
    br, bc = divmod(int(block), NB)
    return [r * GRID + c
            for r in range(br * BLOCK, (br + 1) * BLOCK)
            for c in range(bc * BLOCK, (bc + 1) * BLOCK)]


def border_blocks() -> list[int]:
    """The 28 blocks on the edge of the 8x8 grid."""
    return [b for b in range(NB * NB)
            if b // NB in (0, NB - 1) or b % NB in (0, NB - 1)]


def block_slice(a: np.ndarray, b: int) -> tuple[slice, slice]:
    """Pixel rows and columns of block ``b`` (identical to ``readout_generation_check``)."""
    h, w = a.shape[:2]
    br, bc = divmod(int(b), NB)
    return (slice(int(br * BLOCK * h / GRID), int((br + 1) * BLOCK * h / GRID)),
            slice(int(bc * BLOCK * w / GRID), int((bc + 1) * BLOCK * w / GRID)))


def fill(arr: np.ndarray, b: int, how: str) -> Any:
    """Occlude block ``b``: ``black``, ``mean`` (image mean) or ``blur`` (Gaussian, radius 12)."""
    from PIL import Image, ImageFilter  # noqa: PLC0415

    a = arr.copy()
    ys, xs = block_slice(a, b)
    if how == "black":
        a[ys, xs] = 0
    elif how == "mean":
        a[ys, xs] = int(arr.mean())
    elif how == "blur":
        a[ys, xs] = np.array(Image.fromarray(arr[ys, xs]).filter(ImageFilter.GaussianBlur(12)))
    else:
        raise ValueError(f"unknown fill {how!r}")
    return Image.fromarray(a)


def draw_films(
    pool: Sequence[tuple[str, str, str]],
    groups: dict[str, Callable[[str, str], bool]],
    accept: Callable[[str, str], bool],
    *,
    n: int,
    used: set[int],
    max_scan: int,
) -> dict[str, list[str]]:
    """Draw up to ``n`` films per group, one film per patient, no patient in two groups.

    ``pool`` is ``(image, labels, view)`` in the order to visit (already permuted). Each group
    has an eligibility test on ``(labels, view)`` and shares the model test ``accept(group,
    image)``, which is only called on eligible films (it runs the model). Groups take turns, one
    accepted film at a time, so none gets first pick of patients carrying several labels. A
    group stops after ``max_scan`` eligible candidates. ``used`` (patient ids) is updated.
    """
    picks: dict[str, list[str]] = {g: [] for g in groups}
    cursor = dict.fromkeys(groups, 0)
    scanned = dict.fromkeys(groups, 0)
    live = list(groups)
    while live:
        for g in list(live):
            got = False
            while cursor[g] < len(pool) and scanned[g] < max_scan and len(picks[g]) < n:
                name, labels, view = pool[cursor[g]]
                cursor[g] += 1
                if patient(name) in used or not groups[g](labels, view):
                    continue
                scanned[g] += 1
                if accept(g, name):
                    used.add(patient(name))
                    picks[g].append(name)
                    got = True
                    break
            if not got or len(picks[g]) >= n:
                live.remove(g)
    return picks


class AnswerRunner:
    """MedGemma with the CLT loaded: readouts, answers and feature captures for one film."""

    def __init__(self, cmg: Any, layers: list[int]) -> None:
        from tracecxr.models._base import resolve_yes_no_token_ids  # noqa: PLC0415

        self.cmg = cmg
        self.layers = layers
        st = cmg._state
        self.torch = st["torch"]
        self.proc, self.model, self.clt = st["proc"], st["model"], st["clt"]
        self.tok = getattr(self.proc, "tokenizer", self.proc)
        self.yes_ids, self.no_ids = resolve_yes_no_token_ids(self.tok)
        self.img_tok = int(self.model.config.image_token_index)

    @classmethod
    def load(cls, results_dir: Path, ckpt: str, device: str | None = None) -> AnswerRunner:
        """Load MedGemma 1.5 and the CLT exactly as E2 does."""
        import torch  # noqa: PLC0415

        from tracecxr.core.config import MODELS  # noqa: PLC0415
        from tracecxr.intervention import ClampedMedGemma  # noqa: PLC0415
        from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        res = json.loads((results_dir / "clt_live_result.json").read_text())
        layers = list(res["layers"])
        cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                        k=res.get("k", 32), n_layers=len(layers),
                        adam_8bit=bool(res.get("adam_8bit", False)), device=dev)
        cmg = ClampedMedGemma(ckpt, layers, clt_config=cfg, model_id=MODELS["medgemma"].locator,
                              device=dev)
        cmg._load()
        return cls(cmg, layers)

    # ---- readouts --------------------------------------------------------------------------
    def readouts(self, row: Any) -> dict[str, Any]:
        """P_raw, P_norm, answer mass, yes-minus-no logit gap and the top-10 next tokens."""
        from tracecxr.models._base import yes_probability  # noqa: PLC0415

        row = row.float()
        pv = self.torch.softmax(row, -1)
        y, n = pv[self.yes_ids].max(), pv[self.no_ids].max()
        top = self.torch.topk(pv, 10)
        return {"raw": float(y), "norm": float(yes_probability(row, self.yes_ids, self.no_ids)),
                "mass": float(y + n),
                "gap": float(row[self.yes_ids].max() - row[self.no_ids].max()),
                "top10": [[int(i), self.tok.decode([int(i)]), float(p)]
                          for p, i in zip(top.values, top.indices, strict=True)]}

    def _inputs(self, prompt: str, image: Any) -> Any:
        return self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(self.model.device)

    def plain_row(self, prompt: str, image: Any) -> Any:
        """Decision-position logits of the plain model (equal to the CLT path with no edits)."""
        inp = self._inputs(prompt, image)
        with self.torch.inference_mode():
            return self.model(**inp).logits[0, -1].float()

    def plain_greedy(self, prompt: str, image: Any, max_new_tokens: int) -> str:
        """The plain model's greedy answer (used to draw films)."""
        inp = self._inputs(prompt, image)
        plen = inp["input_ids"].shape[1]
        with self.torch.inference_mode():
            g = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                    top_k=None, top_p=None)
        return self.tok.decode(g[0, plen:], skip_special_tokens=True)

    def answer(self, prompt: str, image: Any, edits: Sequence[Any] = (), *, n_samples: int,
               max_new_tokens: int, tol: float = 0.02, mode: str = "replace") -> dict[str, Any]:
        """Readouts, greedy answer and ``n_samples`` T=1 samples under ``edits`` (E2's run_arm).

        Hard gate: the first generated token's distribution must give the same P_raw and P_norm
        as the scored forward, within ``tol``. ``mode`` is the clamp's (``"replace"``, as in
        E2-E5, or ``"add"``: the edit's change added to the live MLPs).
        """
        scored = self.cmg.decision_logits_row(prompt=prompt, image=image, edits=list(edits),
                                              mode=mode)
        gen = self.cmg.generate(prompt=prompt, image=image, edits=list(edits),
                                max_new_tokens=max_new_tokens, num_samples=n_samples, mode=mode)
        ro, first = self.readouts(scored), self.readouts(gen["first_logits"])
        diff = max(abs(ro["raw"] - first["raw"]), abs(ro["norm"] - first["norm"]))
        if diff >= tol:
            raise RuntimeError(f"generation prefill does not match the scored forward ({diff:.4f})")
        return {**ro, "first_token_diff": diff, "greedy": gen["greedy"],
                "greedy_parsed": parse_answer(gen["greedy"]), "samples": gen["samples"],
                "samples_parsed": [parse_answer(t) for t in gen["samples"]],
                "samples_open": [opening_word(t) for t in gen["samples"]]}

    # ---- features --------------------------------------------------------------------------
    def image_feats(self, prompt: str, image: Any) -> np.ndarray:
        """CLT activations at the 256 image tokens, ``(256, n_layers, n_features)`` float32.

        Same capture as E2's ``mean_feats`` (the whole sequence is encoded, then image columns
        are taken), so ``image_feats(...).mean(0)`` reproduces it.
        """
        torch = self.torch
        inputs = self._inputs(prompt, image)
        cap: dict[int, Any] = {}

        def mk(L: int):  # noqa: ANN202
            def hook(_m, inp, _o):  # noqa: ANN001, ANN202
                cap[L] = inp[0][0]
            return hook

        mods = self.cmg._state["layers_mod"]
        hs = [mods[L].mlp.register_forward_hook(mk(L)) for L in self.layers]
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hs:
                h.remove()
        cols = (inputs["input_ids"][0] == self.img_tok).nonzero(as_tuple=True)[0]
        mlp_in = torch.stack([cap[L].float() for L in self.layers], dim=1)
        with torch.no_grad():
            f = self.clt.encode(mlp_in)
        return f[cols].cpu().numpy().astype(np.float32)

    def topn(self, score: np.ndarray, n: int) -> list[tuple[int, int]]:
        """Global top-``n`` (layer, feature) pairs of a ``(n_layers, n_features)`` score."""
        flat = np.argsort(-score, axis=None)[:n]
        return [(self.layers[int(i // score.shape[1])], int(i % score.shape[1])) for i in flat]

    @staticmethod
    def edits(pairs: Iterable[tuple[int, int]], op: str, amount: float) -> list[Any]:
        """Clamp edits at image positions, as in E2."""
        from tracecxr.intervention import FeatureEdit  # noqa: PLC0415

        return [FeatureEdit(layer=int(L), feature=int(f), op=op, amount=amount,
                            positions="image") for L, f in pairs]

    def scan_blocks(self, prompt: str, arr: np.ndarray) -> list[float]:
        """Yes-minus-no logit gap with each of the 64 blocks filled black (plain model)."""
        out = []
        for b in range(NB * NB):
            row = self.plain_row(prompt, fill(arr, b, "black"))
            out.append(float(row[self.yes_ids].max() - row[self.no_ids].max()))
        return out
