"""CLT feature clamping on MedGemma — the Stage 4 causal-intervention engine.

Where :mod:`tracecxr.intervention.patching` is the abstract constrained-patching contract
(over the generic :class:`~tracecxr.core.ModelAdapter`), this module is the concrete engine
that actually reaches individual cross-layer-transcoder features inside MedGemma. It runs
the local replacement model (each captured decoder MLP output swapped for the CLT's
reconstruction + a frozen per-position error) and lets you *edit* feature activations before
they are written back — so you can suppress the report-prior features and watch a finding
collapse, or amplify them on a clean image to manufacture a hallucination (the bidirectional
"dial").

Design notes:
- The forward is run twice: once clean to capture every captured layer's MLP input/output
  (hence the CLT features and the frozen error), then once with the edited reconstruction
  substituted in. With no edits the substitution reproduces the original logits exactly, so
  the no-edit arm is a faithful baseline — the clamp-vs-no-clamp delta isolates the edit.
- It deliberately substitutes *all* captured layers in one pass rather than the paper's
  "resume from the last layer of the range" optimization: over a 12-layer window that
  optimization buys little and the single-pass form is simpler and harder to get wrong.
- The manifesto's guidance is to **negative-steer** cross-layer features (scale by ``-1``)
  rather than zero-ablate; :class:`FeatureEdit` defaults to ``op="scale", amount=-1.0``.

The pure feature-edit core (:func:`apply_feature_edits`) is unit-tested in CI; the runtime
(:class:`ClampedMedGemma`) needs MedGemma + a CLT checkpoint on a GPU and is ``requires_gpu``.
Torch is lazy-imported so this module imports offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tracecxr.transcoder.clt import CLTConfig, CrossLayerTranscoder

_OPS = ("scale", "ablate", "set")
_POSITIONS = ("all", "image", "text")
_MODES = ("replace", "add")


@dataclass(frozen=True)
class FeatureEdit:
    """One edit to a CLT feature's activation (manifesto §Stage 4).

    Attributes:
        layer: model decoder layer the feature lives at (must be one the CLT was trained on).
        feature: feature index within that layer's dictionary.
        op: ``"scale"`` (multiply by ``amount``), ``"ablate"`` (zero it), or ``"set"`` (force
            to ``amount``).
        amount: the scale factor or set value. Default ``-1.0`` = negative steering (the
            manifesto's recommended suppression); a positive ``>1`` amplifies (induce a
            hallucination on a clean case).
        positions: which token positions to edit — ``"all"``, ``"image"`` (image patches
            only), or ``"text"`` (prompt/context only). ``"image"``/``"text"`` need an image
            mask, supplied by the runtime.
    """

    layer: int
    feature: int
    op: str = "scale"
    amount: float = -1.0
    positions: str = "all"

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"op must be one of {_OPS}, got {self.op!r}")
        if self.positions not in _POSITIONS:
            raise ValueError(f"positions must be one of {_POSITIONS}, got {self.positions!r}")


def apply_feature_edits(
    feats: Any,
    edits: list[FeatureEdit],
    layers: list[int],
    *,
    img_mask: Any | None = None,
) -> Any:
    """Apply ``edits`` to a ``(seq, n_layers, n_features)`` feature-activation array.

    Backend-agnostic: works on a NumPy array or a torch tensor (only ``.copy``/``.clone``,
    boolean-mask indexing, and elementwise arithmetic are used). Returns a new array; the
    input is not mutated.

    Args:
        feats: ``(seq, n_layers, n_features)`` CLT activations.
        edits: the edits to apply, in order.
        layers: model layers in CLT index order (maps ``edit.layer`` → the ``n_layers`` axis).
        img_mask: boolean ``(seq,)`` mask, ``True`` at image-patch positions — required only
            if any edit restricts to ``"image"``/``"text"`` positions.

    Returns:
        The edited activations (same type/shape as ``feats``).

    Raises:
        KeyError: if an edit targets a layer the CLT was not trained on.
        ValueError: if a position-restricted edit is given without ``img_mask``.
    """
    out = feats.copy() if hasattr(feats, "copy") else feats.clone()
    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    for e in edits:
        if e.layer not in layer_to_idx:
            raise KeyError(
                f"edit targets model layer {e.layer}, not in the CLT's layers {layers}"
            )
        idx = layer_to_idx[e.layer]
        if e.positions == "all":
            sel: Any = slice(None)
        else:
            if img_mask is None:
                raise ValueError(f"positions={e.positions!r} needs an img_mask")
            sel = img_mask if e.positions == "image" else ~img_mask
        target = out[sel, idx, e.feature]
        if e.op == "ablate":
            new = target * 0
        elif e.op == "scale":
            new = target * e.amount
        else:  # "set"
            new = target * 0 + e.amount
        out[sel, idx, e.feature] = new
    return out


def prefill_substitute_hook(recon_layer: Any, prompt_len: int) -> Any:
    """MLP forward hook that swaps in the replacement-model output on the prefill pass only.

    ``recon_layer`` is one layer's ``(prompt_len, d_model)`` reconstruction. On a forward whose
    sequence length is ``prompt_len`` (the single decision pass, or the prefill of a generate
    call) the MLP output is replaced, broadcast over the batch so ``num_return_sequences > 1``
    works. Decode steps (sequence length 1) pass through untouched: generated tokens run the
    live MLPs and attend to the edited prompt through the KV cache.
    """

    def hook(_m, _inp, out):  # noqa: ANN001, ANN202
        ref = out[0] if isinstance(out, tuple) else out
        if ref.shape[1] != prompt_len:
            return None
        return recon_layer.unsqueeze(0).expand(ref.shape[0], -1, -1).to(ref.dtype)

    return hook


def prefill_add_hook(delta_layer: Any, prompt_len: int) -> Any:
    """MLP forward hook that adds an edit's change to the live MLP output, on the prefill only.

    ``delta_layer`` is one layer's ``(prompt_len, d_model)`` change ``decode(edited) -
    decode(clean)``, zero wherever no feature was edited. Unlike :func:`prefill_substitute_hook`
    the MLP keeps computing from its live input everywhere, so positions the edit reaches through
    attention (e.g. the readout position) respond to it. Decode steps pass through untouched.
    """

    def hook(_m, _inp, out):  # noqa: ANN001, ANN202
        ref = out[0] if isinstance(out, tuple) else out
        if ref.shape[1] != prompt_len:
            return None
        return ref + delta_layer.unsqueeze(0).to(ref.dtype)

    return hook


def _prefill_hook(mode: str) -> Any:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    return prefill_substitute_hook if mode == "replace" else prefill_add_hook


class ClampedMedGemma:
    """Run MedGemma with CLT feature edits applied to the replacement model (manifesto §Stage 4).

    Args:
        clt_checkpoint: path to the trained CLT checkpoint.
        layers: model decoder layers the CLT was trained on, in CLT index order.
        clt_config: config matching the trained CLT (n_features/span/k/n_layers) so the
            checkpoint loads.
        model_id: MedGemma locator override (defaults to the registry entry).
        device: torch device override.
    """

    def __init__(
        self,
        clt_checkpoint: str,
        layers: list[int],
        *,
        clt_config: CLTConfig | None = None,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.clt_checkpoint = clt_checkpoint
        self.layers = list(layers)
        self.clt_config = clt_config
        self.model_id = model_id
        self.device = device
        self._state: dict[str, Any] = {}

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        from tracecxr.core.config import MODELS

        mid = self.model_id or MODELS["medgemma"].locator
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype).to(dev).eval()
        clt = CrossLayerTranscoder(self.clt_config or CLTConfig(device=dev))
        clt.load_checkpoint(self.clt_checkpoint, load_optimizer=False)  # inference only (OOM)
        self._state.update(
            torch=torch, proc=proc, model=model, dev=dev, clt=clt,
            img_id=int(model.config.image_token_index),
            layers_mod=model.model.language_model.layers,
        )

    def _decision_logits(
        self, prompt: str, image: Any | None, edits: list[FeatureEdit],
        *, messages: list[dict[str, Any]] | None = None, mode: str = "replace",
    ) -> Any:
        """Logits at the generation position under ``edits`` (empty -> faithful baseline).

        Pass ``messages`` (a full chat-template conversation) instead of ``prompt``/``image`` to
        clamp inside a multi-turn context, e.g. the reason-then-probe hallucination framing
        (user interpret -> assistant reasoning -> user yes/no probe). When given it takes
        precedence and the single-turn ``prompt``/``image`` are ignored.
        """
        hook = _prefill_hook(mode)
        inputs, recon = self._prepare(prompt, image, edits, messages=messages, mode=mode)
        torch = self._state["torch"]
        model, layers_mod = self._state["model"], self._state["layers_mod"]
        decision = recon.shape[0] - 1
        sub_handles = [
            layers_mod[L].mlp.register_forward_hook(hook(recon[:, i], recon.shape[0]))
            for i, L in enumerate(self.layers)
        ]
        try:
            with torch.inference_mode():
                out = model(**inputs)
        finally:
            for h in sub_handles:
                h.remove()
        return out.logits[0, decision].float()

    def _prepare(
        self, prompt: str, image: Any | None, edits: list[FeatureEdit],
        *, messages: list[dict[str, Any]] | None = None, mode: str = "replace",
    ) -> tuple[Any, Any]:
        """Tokenize, run the clean capture pass, and return ``(inputs, recon)``.

        ``mode="replace"``: ``recon`` is the ``(prompt_len, n_layers, d_model)`` replacement-model
        MLP output under ``edits``: CLT decode of the edited features plus the frozen per-position
        error, so ``edits=[]`` reproduces the clean MLP outputs. The features are encoded once, on
        the clean pass, so at positions with no edit the substituted output is the clean one.
        ``mode="add"``: ``recon`` is the change ``decode(edited) - decode(clean)``, to be added to
        the live MLP outputs (zero for ``edits=[]``).
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if not self._state:
            self._load()
        torch = self._state["torch"]
        proc, model, dev, clt = (self._state[k] for k in ("proc", "model", "dev", "clt"))

        if messages is None:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if image is not None:
                from PIL import Image  # noqa: PLC0415

                pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
                content.insert(0, {"type": "image", "image": pil.convert("RGB")})
            messages = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(dev)
        ids = inputs["input_ids"][0]
        img_mask = ids == self._state["img_id"]
        layers_mod = self._state["layers_mod"]

        # phase 1: clean capture of each captured layer's MLP input/output.
        cap: dict[int, tuple[Any, Any]] = {}

        def mk(layer: int):
            def hook(_m, inp, out):  # noqa: ANN001
                oo = out[0] if isinstance(out, tuple) else out
                cap[layer] = (inp[0][0], oo[0])
            return hook

        handles = [layers_mod[L].mlp.register_forward_hook(mk(L)) for L in self.layers]
        try:
            with torch.inference_mode():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()

        with torch.inference_mode():
            mlp_in = torch.stack([cap[L][0].float() for L in self.layers], dim=1)
            mlp_out = torch.stack([cap[L][1].float() for L in self.layers], dim=1)
            feats = clt.encode(mlp_in)
            error = mlp_out - clt.decode(feats)
            edited = apply_feature_edits(feats, edits, self.layers, img_mask=img_mask)
            if mode == "add":
                return inputs, clt.decode(edited) - clt.decode(feats)
            recon = clt.decode(edited) + error  # edits=[] -> reproduces mlp_out exactly
        return inputs, recon

    def generate(
        self,
        *,
        prompt: str | None = None,
        image: Any | None = None,
        edits: list[FeatureEdit] | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 64,
        num_samples: int = 0,
        mode: str = "replace",
    ) -> dict[str, Any]:
        """Free generation under ``edits``: the clamp is applied on the prompt, then decoding runs.

        The prefill pass is exactly the forward :meth:`decision_logits_row` scores, so the first
        generated token is drawn from the same distribution the readouts are computed from
        (returned as ``first_logits`` so a caller can assert it). Decode steps run the live
        model on the generated tokens, attending to the edited prompt through the KV cache.

        Returns:
            ``{"greedy": str, "samples": list[str], "first_logits": tensor}``. ``samples`` holds
            ``num_samples`` draws at temperature 1 with no top-k/top-p truncation, so they are
            samples from the model's own distribution.
        """
        hook = _prefill_hook(mode)
        inputs, recon = self._prepare(prompt or "", image, list(edits or []), messages=messages,
                                      mode=mode)
        torch = self._state["torch"]
        model, layers_mod = self._state["model"], self._state["layers_mod"]
        tok = getattr(self._state["proc"], "tokenizer", self._state["proc"])
        plen = recon.shape[0]
        handles = [
            layers_mod[L].mlp.register_forward_hook(hook(recon[:, i], plen))
            for i, L in enumerate(self.layers)
        ]
        try:
            with torch.inference_mode():
                g = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   top_k=None, top_p=None, output_logits=True,
                                   return_dict_in_generate=True)
                samples: list[str] = []
                if num_samples:
                    s = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                       temperature=1.0, top_k=0, top_p=1.0,
                                       num_return_sequences=num_samples)
                    samples = tok.batch_decode(s[:, plen:], skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()
        return {"greedy": tok.decode(g.sequences[0, plen:], skip_special_tokens=True),
                "samples": samples, "first_logits": g.logits[0][0].float()}

    def answer_token_id(self, prompt: str, *, answer: str = "yes", image: Any | None = None) -> int:
        """Resolve ``answer`` to the token id the model actually uses on the clean forward.

        Resolve once from the unperturbed distribution and reuse the id across clamp arms so
        the clamp-vs-no-clamp comparison tracks the *same* token (see
        :func:`~tracecxr.attribution.medgemma_backend.best_answer_token_id`).
        """
        from tracecxr.attribution.medgemma_backend import best_answer_token_id  # noqa: PLC0415

        if not self._state:
            self._load()
        torch = self._state["torch"]
        logits = self._decision_logits(prompt, image, [])
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        tok = getattr(self._state["proc"], "tokenizer", self._state["proc"])
        return best_answer_token_id(tok, answer, probs)

    def finding_probability(
        self,
        prompt: str,
        *,
        answer: str = "yes",
        image: Any | None = None,
        edits: list[FeatureEdit] | None = None,
        answer_id: int | None = None,
    ) -> float:
        """Probability of the ``answer`` token at the decision position under ``edits``.

        With ``edits=None`` this is the faithful baseline (the substitution reproduces the
        original logits). Pass a clamp to read the intervened probability; the clamp-vs-no-
        clamp delta is the causal effect. Pass ``answer_id`` (from :meth:`answer_token_id`) to
        pin the target token across arms; otherwise it is resolved from this call's
        distribution.
        """
        from tracecxr.attribution.medgemma_backend import best_answer_token_id  # noqa: PLC0415

        if not self._state:
            self._load()
        torch = self._state["torch"]
        logits = self._decision_logits(prompt, image, list(edits or []))
        probs = torch.softmax(logits, dim=-1)
        if answer_id is None:
            tok = getattr(self._state["proc"], "tokenizer", self._state["proc"])
            answer_id = best_answer_token_id(tok, answer, probs.detach().cpu().numpy())
        return float(probs[answer_id].detach().cpu())

    def decision_logits_row(
        self,
        *,
        prompt: str | None = None,
        image: Any | None = None,
        edits: list[FeatureEdit] | None = None,
        messages: list[dict[str, Any]] | None = None,
        mode: str = "replace",
    ) -> Any:
        """The full logits row at the decision position under ``edits`` (a torch 1-D tensor).

        Use this when the readout is a multi-token function of the distribution (e.g. the 2-way
        ``yes_probability`` probe), rather than a single answer token. Pass ``messages`` for a
        multi-turn context (reason-then-probe); otherwise ``prompt``/``image`` build a single turn.
        """
        return self._decision_logits(prompt or "", image, list(edits or []), messages=messages,
                                     mode=mode)

    def sweep(
        self,
        prompt: str,
        features: list[tuple[int, int]],
        factors: list[float],
        *,
        answer: str = "yes",
        image: Any | None = None,
        positions: str = "all",
        answer_id: int | None = None,
    ) -> list[tuple[float, float]]:
        """The hallucination dial: ``answer`` probability vs a scale ``factor`` on ``features``.

        For each factor, every ``(layer, feature)`` in ``features`` is scaled by it (``0`` =
        ablate, ``1`` = no-op, ``>1`` = amplify), then the ``answer`` probability is measured.
        Sweeping below and above 1 gives the bidirectional curve — suppress to cure, amplify
        to induce.

        Returns:
            ``[(factor, probability), …]`` in the order of ``factors``.
        """
        curve: list[tuple[float, float]] = []
        for fac in factors:
            edits = [
                FeatureEdit(layer=layer, feature=feat, op="scale", amount=fac, positions=positions)
                for layer, feat in features
            ]
            curve.append((fac, self.finding_probability(prompt, answer=answer, image=image,
                                                        edits=edits, answer_id=answer_id)))
        return curve
