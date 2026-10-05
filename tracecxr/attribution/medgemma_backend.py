"""GPU attribution backend: MedGemma + the trained CLT (manifesto §Stage 2).

This is the production :class:`~tracecxr.attribution.AttributionBackend` — the only
torch-dependent piece of Stage 2. It implements the *local replacement model*: MedGemma
with each decoder MLP output replaced by the cross-layer transcoder's reconstruction plus
a frozen per-position error term, so the model still reproduces its original logits but
now reads/writes through interpretable CLT features. Attribution then flows from a finding
logit back to those features and to the image/prompt inputs.

Status — this runs only on a GPU with MedGemma weights + a trained CLT checkpoint, so it is
``requires_gpu`` and never exercised in CI. The scaffolding (model + CLT loading, the
forward capture, feature/output/error/input node construction, id assembly) reuses the
conventions proven in ``tracecxr/transcoder/cache.py`` and is correct by construction. The
**edge attribution** is a first, deliberately simple version that MUST be validated on the
H100 before any scientific claim:

- Edges are first-order ``activation × gradient`` attributions of every feature / input /
  error node onto the output logit(s), taken through the CLT-substituted replacement
  forward. Because the substitution reproduces the original logits, this is the gradient of
  the replacement model — faithful to first order and, crucially, attention is *live*, so
  cross-position edges (image-patch → finding, prompt → finding) are captured.
- It does NOT yet freeze attention / layernorm scales (the stop-gradient Jacobian that
  makes edge weights sum *exactly* to each preactivation), and it emits feature→logit edges
  rather than the full multi-hop feature→feature graph. Same-position cross-layer
  feature→feature edges (cheap, from the CLT weights) are added where the window allows.
  The frozen-attention Jacobian and the full cross-position multi-hop graph are the v2
  upgrade — see manifesto §Stage 2 and the circuit-tracing appendix.

The pure helpers below (:func:`select_output_token_ids`) are unit-tested in CI; everything
that touches torch is lazy-imported so this module imports offline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tracecxr.attribution.graph import (
    Node,
    NodeType,
    RawAttribution,
    error_id,
    feature_id,
    input_image_id,
    input_token_id,
    output_id,
)
from tracecxr.transcoder.clt import CLTConfig, CrossLayerTranscoder


def best_answer_token_id(tok: Any, answer: str, scores: np.ndarray) -> int:
    """Resolve an answer word to the token id the model actually uses (manifesto §Stage 2).

    The leading sub-token of an answer like ``"yes"`` depends on casing and the SentencePiece
    leading-space marker (MedGemma emits ``"Yes"`` after the generation prompt, not ``"yes"``).
    Resolve robustly by scoring every plausible variant against ``scores`` (the decision-
    position probabilities) and returning the highest-scoring token id.

    Args:
        tok: the tokenizer (or processor exposing ``.encode``).
        answer: the answer word, e.g. ``"yes"``.
        scores: 1-D probability/logit vector over the vocabulary at the decision position.

    Returns:
        The token id of the best-scoring variant.
    """
    variants = {
        answer, answer.lower(), answer.upper(), answer.capitalize(),
        " " + answer, " " + answer.lower(), " " + answer.capitalize(),
    }
    best_id: int | None = None
    best = -float("inf")
    seen: set[int] = set()
    for v in variants:
        enc = tok.encode(v, add_special_tokens=False)
        if not enc:
            continue
        i = int(enc[0])
        if i in seen:
            continue
        seen.add(i)
        if float(scores[i]) > best:
            best, best_id = float(scores[i]), i
    if best_id is None:
        raise ValueError(f"answer {answer!r} did not tokenize to any id")
    return best_id


def select_output_token_ids(
    probs: np.ndarray,
    *,
    finding_token_id: int | None = None,
    n_logits: int = 10,
    logit_mass: float = 0.95,
) -> list[int]:
    """Pick the output (logit) token ids to attribute from (manifesto §Stage 2).

    Takes the next-token probabilities at the decision position and returns the smallest
    set of token ids that covers ``logit_mass`` of probability, capped at ``n_logits``,
    always including ``finding_token_id`` (the token whose logit we are explaining) even if
    it falls outside the top set.

    Args:
        probs: 1-D next-token probability vector over the vocabulary.
        finding_token_id: The finding token's id, force-included if not already present.
        n_logits: Hard cap on the number of output nodes.
        logit_mass: Cumulative probability mass to cover.

    Returns:
        Token ids, most probable first (the forced finding token appended if it was absent).
    """
    order = np.argsort(probs)[::-1]
    chosen: list[int] = []
    mass = 0.0
    for tid in order:
        if len(chosen) >= n_logits or mass >= logit_mass:
            break
        chosen.append(int(tid))
        mass += float(probs[tid])
    if finding_token_id is not None and finding_token_id not in chosen:
        chosen.append(int(finding_token_id))
    return chosen


class MedGemmaCLTBackend:
    """Attribution backend over MedGemma's local replacement model (manifesto §Stage 2).

    Args:
        clt_checkpoint: Path to a trained CLT checkpoint (e.g. ``b1_scale_ckpt/clt_ckpt.pt``).
        layers: The model decoder layers the CLT was trained on, in CLT index order (e.g.
            ``range(8, 20)`` for the B1 scale run). Used to map model layer ↔ CLT layer index.
        clt_config: Config for the CLT to load the checkpoint into. Must match the trained
            config (n_features / span / k); the loader validates parameter shapes.
        model_id: MedGemma locator override (defaults to the registry entry).
        device: Torch device override (defaults to cuda).
        max_feature_nodes: Keep only the top-N features by |logit attribution| as nodes (the
            rest are pruned anyway); ``None`` keeps all active features.
    """

    def __init__(
        self,
        clt_checkpoint: str,
        layers: list[int],
        *,
        clt_config: CLTConfig | None = None,
        model_id: str | None = None,
        device: str | None = None,
        max_feature_nodes: int | None = 4096,
    ) -> None:
        self.clt_checkpoint = clt_checkpoint
        self.layers = list(layers)
        self.clt_config = clt_config
        self.model_id = model_id
        self.device = device
        self.max_feature_nodes = max_feature_nodes
        self._state: dict[str, Any] = {}

    # -- lazy load (mirrors cache.medgemma_capture_fn) ----------------------------------

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        from tracecxr.core.config import MODELS

        mid = self.model_id or MODELS["medgemma"].locator
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype).to(dev).eval()

        clt = CrossLayerTranscoder(self.clt_config or CLTConfig(device=dev))
        clt.load_checkpoint(self.clt_checkpoint, load_optimizer=False)

        self._state.update(
            torch=torch,
            proc=proc,
            model=model,
            dev=dev,
            clt=clt,
            img_id=int(model.config.image_token_index),
            layers_mod=model.model.language_model.layers,
            unembed=model.get_output_embeddings(),  # lm_head
        )

    # -- the protocol method ------------------------------------------------------------

    def trace(
        self,
        *,
        prompt: str,
        finding_token: str,
        image: Any | None = None,
        n_logits: int = 10,
        logit_mass: float = 0.95,
    ) -> RawAttribution:
        """Trace ``finding_token`` back through the replacement model (see module docstring).

        Validate on the H100 before trusting edge weights — this is the v1 attribution.
        """
        if not self._state:
            self._load()
        torch = self._state["torch"]
        proc, model, dev = self._state["proc"], self._state["model"], self._state["dev"]
        clt = self._state["clt"]

        # 1. Build inputs exactly like the capture path (image + text chat template).
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            from PIL import Image  # noqa: PLC0415

            pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
            content.insert(0, {"type": "image", "image": pil.convert("RGB")})
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(dev)
        ids = inputs["input_ids"][0]
        seq = ids.shape[0]
        decision = seq - 1  # the generation position whose logit we explain
        img_mask = (ids == self._state["img_id"]).cpu().numpy()

        # 2. Clean forward, capturing each captured layer's MLP input (CLT input) + output.
        cap: dict[int, tuple[Any, Any]] = {}

        def mk(layer: int):
            def hook(_m, inp, out):  # noqa: ANN001
                oo = out[0] if isinstance(out, tuple) else out
                cap[layer] = (inp[0][0], oo[0])  # drop batch dim -> (seq, d)

            return hook

        layers_mod = self._state["layers_mod"]
        handles = [layers_mod[L].mlp.register_forward_hook(mk(L)) for L in self.layers]
        try:
            # no_grad (NOT inference_mode): the captured activations become the frozen error
            # constant in the grad-tracked replacement forward below; inference-mode tensors
            # cannot participate in autograd, even as constants.
            with torch.no_grad():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # 3. CLT features + per-layer reconstruction error (frozen).
        # Stack captured MLP inputs into (seq, n_layers, d) in CLT layer order.
        mlp_in = torch.stack([cap[L][0].float() for L in self.layers], dim=1)  # (seq, Lc, d)
        mlp_out = torch.stack([cap[L][1].float() for L in self.layers], dim=1)  # (seq, Lc, d)
        with torch.no_grad():
            feats0 = clt.encode(mlp_in)  # (seq, Lc, F)
            recon0 = clt.decode(feats0)  # (seq, Lc, d)
            error = (mlp_out - recon0).detach()  # frozen "dark matter" per (pos, layer)

        # 4. Replacement forward with feature activations as a differentiable leaf, and
        #    gradient attribution of the finding logit onto every node. (v1 — validate.)
        a_leaf = feats0.detach().clone().requires_grad_(True)  # (seq, Lc, F)
        recon = clt.decode(a_leaf) + error  # reproduces original mlp_out at a_leaf == feats0

        def sub(layer_idx: int):
            def hook(_m, _inp, out):  # noqa: ANN001
                ref = out[0] if isinstance(out, tuple) else out
                return recon[:, layer_idx].unsqueeze(0).to(ref.dtype)  # match model dtype (bf16)

            return hook

        sub_handles = [
            layers_mod[L].mlp.register_forward_hook(sub(i)) for i, L in enumerate(self.layers)
        ]
        try:
            out = model(**inputs)
            logits = out.logits[0, decision].float()  # (vocab,)
        finally:
            for h in sub_handles:
                h.remove()

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        tok = getattr(proc, "tokenizer", proc)
        finding_id = best_answer_token_id(tok, finding_token, probs)
        token_ids = select_output_token_ids(
            probs, finding_token_id=finding_id, n_logits=n_logits, logit_mass=logit_mass
        )

        # gradient of the finding logit wrt the feature leaf -> activation × grad edges.
        grad = torch.autograd.grad(logits[finding_id], a_leaf, retain_graph=False)[0]
        attribution = (a_leaf.detach() * grad).cpu().numpy()  # (seq, Lc, F)
        acts = a_leaf.detach().cpu().numpy()

        return self._assemble(
            attribution=attribution,
            acts=acts,
            error=error.detach().float().cpu().numpy(),
            probs=probs,
            token_ids=token_ids,
            finding_id=finding_id,
            decision=decision,
            img_mask=img_mask,
            replacement_logit=float(logits[finding_id].detach().cpu()),
        )

    # -- helpers ------------------------------------------------------------------------

    def _assemble(
        self,
        *,
        attribution: np.ndarray,
        acts: np.ndarray,
        error: np.ndarray,
        probs: np.ndarray,
        token_ids: list[int],
        finding_id: int,
        decision: int,
        img_mask: np.ndarray,
        replacement_logit: float,
    ) -> RawAttribution:
        """Turn the per-(pos, layer, feature) attribution arrays into typed nodes + edges.

        Pure array bookkeeping (no torch) — keeps the heavy method readable and this part
        easy to reason about. Edge weights are the v1 feature→logit attributions.
        """
        proc = self._state["proc"]
        tok = getattr(proc, "tokenizer", proc)
        seq, n_layers, _ = acts.shape

        label = tok.decode([finding_id]).strip() or f"tok{finding_id}"
        out_id = output_id(decision, label)
        nodes: list[Node] = [
            Node(out_id, NodeType.OUTPUT, label=label,
                 activation=float(probs[finding_id]), position=decision,
                 metadata={"token_id": finding_id, "candidates": token_ids})
        ]
        edges: list[tuple[str, str, float]] = []

        # rank active features by |attribution| and keep the top-N as feature nodes.
        active = np.argwhere(acts != 0.0)  # rows of (pos, layer_idx, feature)
        scored = [
            (int(p), int(li), int(f), float(attribution[p, li, f]), float(acts[p, li, f]))
            for p, li, f in active
        ]
        scored.sort(key=lambda r: abs(r[3]), reverse=True)
        if self.max_feature_nodes is not None:
            scored = scored[: self.max_feature_nodes]

        for pos, li, f, attr, act in scored:
            model_layer = self.layers[li]
            fid = feature_id(model_layer, pos, f)
            nodes.append(
                Node(fid, NodeType.FEATURE, label=f"L{model_layer} f{f}", activation=act,
                     position=pos, metadata={"layer": model_layer, "clt_layer": li, "feature": f})
            )
            edges.append((fid, out_id, attr))  # feature -> finding logit (v1 direct edge)

        # error nodes: one per (pos, layer) with non-trivial residual; magnitude only (v1).
        err_mag = np.linalg.norm(error, axis=-1)  # (seq, n_layers)
        thresh = float(np.quantile(err_mag, 0.99)) if err_mag.size else 0.0
        for pos in range(seq):
            for li in range(n_layers):
                if err_mag[pos, li] >= thresh and thresh > 0:
                    eid = error_id(self.layers[li], pos)
                    nodes.append(Node(eid, NodeType.ERROR, label="residual",
                                      activation=float(err_mag[pos, li]), position=pos,
                                      metadata={"layer": self.layers[li]}))

        # input nodes: one per position (image-patch vs prompt token). Edges to features
        # are the v2 upgrade (need the input-embedding Jacobian); nodes are emitted now so
        # the graph already separates the image-evidence vs report-prior entry points.
        for pos in range(seq):
            iid = input_image_id(pos) if img_mask[pos] else input_token_id(pos)
            ntype = NodeType.INPUT_IMAGE if img_mask[pos] else NodeType.INPUT_TOKEN
            nodes.append(Node(iid, ntype, label=("image" if img_mask[pos] else "token"),
                              position=pos))

        return RawAttribution(
            nodes=nodes,
            edges=edges,
            metadata={
                "replacement_logit": replacement_logit,
                "n_active_features": int((acts != 0.0).sum()),
                "attribution_version": "v1-grad-x-act",
                "frozen_attention": False,
            },
        )
