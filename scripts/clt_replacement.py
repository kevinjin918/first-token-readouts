"""Replacement score: how faithfully the CLT-substituted MedGemma reproduces the original model.

Per Anthropic's circuit-tracing methods ("evaluating graphs — comparing"):
the *local* replacement model (CLT reconstruction + a frozen per-position error term) reproduces the
original logits exactly, so it can't measure faithfulness. The real test drops the error: replace
every MLP's output with ``clt.decode(clt.encode(mlp_in))`` — **no error term** — run the model, and
see how much the prediction moves. High agreement / low KL means the interpretable CLT features (not
the error "dark matter") carry the computation. This is the whole-model counterpart to per-layer
FVU, and the number that matters if you make mechanistic/circuit-level claims (vs. only reducing
hallucinations, where broad-working features matter more than graph faithfulness).

Reports, over a set of prompts (split text vs image):
  * KL(original || replacement) on the decision-position next-token distribution (lower = faithful),
  * top-1 agreement (does the replacement model predict the same token?),
  * finding-token probability recovery (replacement P(finding) / original P(finding)).

Needs a trained checkpoint + GPU. Run after clt_scale.py; pair with clt_eval.py (per-layer FVU) and
the auto-interp pass (interp_collect -> interp_label) for the qualitative "do features make sense?"
half Michael flagged as primary given no clean VLM FVU benchmark exists.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
from tracecxr.attribution.medgemma_backend import MedGemmaCLTBackend, best_answer_token_id
from tracecxr.transcoder.cache import load_manifest
from tracecxr.transcoder.clt import CLTConfig

CACHE = Path.home() / "clt_scale_cache"
CKPT = Path.home() / "clt_scale_ckpt" / "clt_ckpt.pt"
RESULT = Path.home() / "clt_scale_result.json"
CXR_GLOB = str(Path.home() / ".cache/tracecxr/data/chestxray14/images/*.png")
CSV = str(Path.home() / ".cache/tracecxr/data/chexpert_plus/df_chexpert_plus_240401.csv")


def log(*a: object) -> None:
    print(*a, flush=True)


def kl_top1(orig_logits: np.ndarray, repl_logits: np.ndarray) -> tuple[float, bool]:
    """KL(P_orig || P_repl) in nats + whether the argmax token agrees. Pure (testable, no torch)."""
    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    p = softmax(orig_logits.astype(np.float64))
    q = softmax(repl_logits.astype(np.float64))
    kl = float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
    return kl, bool(orig_logits.argmax() == repl_logits.argmax())


def _build_inputs(state: dict[str, Any], text: str, image: Any | None):
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return state["proc"].apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(state["dev"])


def replacement_logits(state: dict[str, Any], layers: list[int], inputs) -> tuple[Any, Any]:
    """Return (original, replacement-no-error) logits at the decision position for one prompt."""
    torch, model, clt = state["torch"], state["model"], state["clt"]
    layers_mod = state["layers_mod"]
    decision = inputs["input_ids"].shape[1] - 1

    cap: dict[int, Any] = {}

    def mk(layer: int):
        def hook(_m, inp, out):  # noqa: ANN001
            cap[layer] = inp[0][0]  # mlp_in, drop batch -> (seq, d)
        return hook

    handles = [layers_mod[L].mlp.register_forward_hook(mk(L)) for L in layers]
    try:
        with torch.no_grad():
            orig = model(**inputs).logits[0, decision].float().detach()
    finally:
        for h in handles:
            h.remove()

    # CLT reconstruction from the clean MLP inputs — NO error term (the faithfulness test).
    mlp_in = torch.stack([cap[L].float() for L in layers], dim=1)  # (seq, Lc, d)
    with torch.no_grad():
        recon = clt.decode(clt.encode(mlp_in))  # (seq, Lc, d)

    def sub(i: int):
        def hook(_m, _inp, out):  # noqa: ANN001
            ref = out[0] if isinstance(out, tuple) else out
            return recon[:, i].unsqueeze(0).to(ref.dtype)
        return hook

    sub_handles = [layers_mod[L].mlp.register_forward_hook(sub(i)) for i, L in enumerate(layers)]
    try:
        with torch.no_grad():
            repl = model(**inputs).logits[0, decision].float().detach()
    finally:
        for h in sub_handles:
            h.remove()
    return orig, repl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--result", default=str(RESULT), help="clt_scale_result.json for the CLT config")
    p.add_argument("--n-text", type=int, default=50, help="report prompts to score")
    p.add_argument("--n-img", type=int, default=50, help="CXR prompts to score")
    p.add_argument("--prompt", default="Interpret this chest X-ray.")
    p.add_argument("--finding", default="Yes", help="finding token for P(finding) recovery")
    p.add_argument("--out", default=str(Path.home() / "clt_replacement.json"))
    args = p.parse_args()

    res = json.loads(Path(args.result).read_text()) if Path(args.result).exists() else {}
    man = load_manifest(args.cache)
    layers = list(man["layers"])
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span"),
                    k=res.get("k", 32), n_layers=len(layers), amp=True,
                    adam_8bit=bool(res.get("adam_8bit", False)))
    backend = MedGemmaCLTBackend(args.ckpt, layers, clt_config=cfg)
    backend._load()
    state = backend._state
    log(f"loaded MedGemma + CLT ({cfg.n_features} feats, span {cfg.span}); scoring replacement")

    import pandas as pd  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    reports = [r for r in pd.read_csv(CSV)["report"].tolist() if isinstance(r, str) and len(r) > 40]
    img_paths = sorted(glob.glob(CXR_GLOB))[: args.n_img]
    jobs = [("text", r[:1500], None) for r in reports[: args.n_text]]
    jobs += [("image", args.prompt, pth) for pth in img_paths]
    if not jobs:
        raise SystemExit("no prompts — check the CheXpert CSV / NIH image paths")

    tok = getattr(state["proc"], "tokenizer", state["proc"])
    by_kind: dict[str, list[tuple[float, bool, float]]] = {"text": [], "image": []}
    for kind, text, pth in jobs:
        image = Image.open(pth).convert("RGB") if pth else None
        inputs = _build_inputs(state, text, image)
        orig, repl = replacement_logits(state, layers, inputs)
        o, r = orig.cpu().numpy(), repl.cpu().numpy()
        kl, agree = kl_top1(o, r)
        fid = best_answer_token_id(tok, args.finding, _softmax_np(o))
        recovery = float(_softmax_np(r)[fid] / max(_softmax_np(o)[fid], 1e-9))
        by_kind[kind].append((kl, agree, recovery))

    report: dict[str, Any] = {"ckpt": args.ckpt, "n_layers": len(layers),
                              "n_features": cfg.n_features, "span": cfg.span}
    for kind, rows in by_kind.items():
        if not rows:
            continue
        kls = [x[0] for x in rows]
        report[kind] = {
            "n": len(rows),
            "kl_mean": float(np.mean(kls)), "kl_median": float(np.median(kls)),
            "top1_agreement": float(np.mean([x[1] for x in rows])),
            "finding_prob_recovery": float(np.mean([x[2] for x in rows])),
        }
        m = report[kind]
        log(f"  {kind:5s} n={m['n']:3d}  KL {m['kl_mean']:.3f} (med {m['kl_median']:.3f})  "
            f"agree {m['top1_agreement']:.2f}  P(find) recov {m['finding_prob_recovery']:.2f}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    log(f"saved {args.out}")


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64) - z.max()
    e = np.exp(z)
    return e / e.sum()


if __name__ == "__main__":
    main()
