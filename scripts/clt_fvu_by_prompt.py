"""Is the off-distribution failure a RECONSTRUCTION failure or an ERROR-COMPOUNDING failure?

scripts/clt_metric_transfer.py showed the CLT-substituted model scores top-1 agreement 1.00 /
KL 0.062 on the single prompt the dictionary was trained on, and 0.00 / KL 13.5 on the constrained
yes/no probes every mechanistic claim in this project is measured with. That was a BEHAVIOURAL
measurement (replacement, error term dropped). It does not say whether the dictionary can still
RECONSTRUCT those activations.

The distinction decides what to do about it:

  RECONSTRUCTION FAILURE   FVU rises off-distribution -> the dictionary genuinely lacks features for
                           those activations. Retraining on the probe distribution is the right fix.

  COMPOUNDING FAILURE      FVU stays flat but replacement still collapses -> the dictionary
                           represents the activations fine, and what is brittle is the
                           drop-the-error-term substitution, where small per-layer errors compound
                           over 34 layers. RETRAINING WOULD NOT FIX THIS, and the honest claim is
                           about the replacement metric rather than about the dictionary.

PRE-SPECIFIED, CORRECTED. The first version keyed on IMAGE-token FVU, which is vacuous: image
tokens precede the text prompt in the sequence, so under causal masking their activations cannot
depend on which prompt follows, and image FVU is identical across prompts by construction (measured:
0.355 for all five). The decision is read at the last TEXT token, so text-token FVU is the metric
that matters.
  fvu_text_probe > 1.5 * fvu_text_train  -> reconstruction failure; retraining is justified.
  fvu_text_probe < 1.2 * fvu_text_train  -> compounding failure; stop the retrain,
                                            reframe the claim.
  between                                -> mixed; report both and claim neither cleanly.
An FVU above 1.0 is worse than predicting the mean, i.e. no useful reconstruction at all.

GPU. Usage (VM):
  python scripts/clt_fvu_by_prompt.py --ckpt /mnt/big/ckpt_live/clt_ckpt.pt \
    --result ~/clt_live_result.json --n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROMPTS = [
    ("train_prompt", "Interpret this chest X-ray."),
    ("open_paraphrase", "Describe in detail all findings visible in this chest radiograph."),
    ("probe_cardio", "Does this chest X-ray show cardiomegaly? Answer yes or no."),
    ("probe_effusion", "Does this chest X-ray show a pleural effusion? Answer yes or no."),
    ("probe_trivial", "Is this image a chest X-ray? Answer yes or no."),
]


def log(*a: object) -> None:
    print(*a, flush=True)


def main() -> None:

    from tracecxr.attribution.medgemma_backend import MedGemmaCLTBackend  # noqa: PLC0415
    from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/mnt/big/ckpt_live/clt_ckpt.pt")
    ap.add_argument("--result", default=str(Path.home() / "clt_live_result.json"))
    ap.add_argument("--banked", default="/mnt/big/attnpatch/attn_causal_map.json")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="/mnt/big/attnpatch/clt_fvu_by_prompt.json")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    layers = list(res["layers"])
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                    k=res.get("k", 32), n_layers=len(layers), amp=True,
                    adam_8bit=bool(res.get("adam_8bit", False)))
    backend = MedGemmaCLTBackend(args.ckpt, layers, clt_config=cfg)
    backend._load()
    state = backend._state
    torch_, model, clt = state["torch"], state["model"], state["clt"]
    proc = state["proc"]
    layers_mod = state["layers_mod"]
    img_tok = int(model.config.image_token_index)
    log(f"loaded MedGemma + CLT ({cfg.n_features} feats, span {cfg.span})")

    def fvu_for(image, prompt):
        """Per-layer FVU of the CLT reconstruction, split into image and text token positions."""
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(model.device)
        cap: dict[int, object] = {}

        def mk(layer):
            def hook(_m, inp, _out):  # noqa: ANN001
                cap[layer] = inp[0][0]
            return hook

        handles = [layers_mod[L].mlp.register_forward_hook(mk(L)) for L in layers]
        try:
            with torch_.no_grad():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # target is the MLP OUTPUT, which is what the transcoder predicts
        out_cap: dict[int, object] = {}

        def mko(layer):
            def hook(_m, _inp, out):  # noqa: ANN001
                out_cap[layer] = (out[0] if isinstance(out, tuple) else out)[0]
            return hook

        h2 = [layers_mod[L].mlp.register_forward_hook(mko(L)) for L in layers]
        try:
            with torch_.no_grad():
                model(**inputs)
        finally:
            for h in h2:
                h.remove()

        mlp_in = torch_.stack([cap[L].float() for L in layers], dim=1)      # (seq, L, d)
        tgt = torch_.stack([out_cap[L].float() for L in layers], dim=1)     # (seq, L, d)
        with torch_.no_grad():
            recon = clt.decode(clt.encode(mlp_in))

        ids = inputs["input_ids"][0]
        is_img = (ids == img_tok)
        res_ = {}
        for name, mask in (("image", is_img), ("text", ~is_img)):
            if mask.sum() == 0:
                continue
            t_, r_ = tgt[mask], recon[mask]
            num = ((t_ - r_) ** 2).sum(-1)
            den = ((t_ - t_.mean(0, keepdim=True)) ** 2).sum(-1).clamp_min(1e-9)
            res_[name] = float((num / den).mean())
        return res_

    banked = json.loads(Path(args.banked).read_text())["A"]["records"][: args.n]
    img_dir = args.data_dir / "images"
    out: dict[str, list] = {n: [] for n, _ in PROMPTS}
    for b in banked:
        path = img_dir / b["image"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        for name, prompt in PROMPTS:
            out[name].append(fvu_for(img, prompt))
        log("  " + b["image"] + "  " + "  ".join(
            f"{n}: img {out[n][-1]['image']:.3f}" for n, _ in PROMPTS))

    log(f"\n=== n={len(out['train_prompt'])} images ===")
    log(f"  {'prompt':18s} {'image FVU':>10s} {'text FVU':>10s}")
    means = {}
    for name, _ in PROMPTS:
        im = float(np.mean([r["image"] for r in out[name]]))
        tx = float(np.mean([r["text"] for r in out[name] if "text" in r]))
        means[name] = {"image": im, "text": tx}
        log(f"  {name:18s} {im:>10.3f} {tx:>10.3f}")

    # TEXT tokens: image FVU cannot vary by prompt (causal masking), so it is not diagnostic.
    ftrain = means["train_prompt"]["text"]
    fprobe = float(np.mean([means[n]["text"] for n, _ in PROMPTS if n.startswith("probe")]))
    ratio = fprobe / max(ftrain, 1e-9)
    log(f"\n  text FVU: training prompt {ftrain:.3f}, probes {fprobe:.3f}, "
        f"ratio {ratio:.2f}x  (TEXT tokens)")
    if ratio > 1.5:
        verdict = ("RECONSTRUCTION FAILURE: the dictionary does not cover the probe activations. "
                   "Retraining on the probe distribution is the right fix.")
    elif ratio < 1.2:
        verdict = ("COMPOUNDING FAILURE: reconstruction is essentially unchanged off-distribution, "
                   "so the dictionary represents these activations fine. The brittleness is in the "
                   "drop-the-error-term replacement test. RETRAINING WILL NOT FIX THIS.")
    else:
        verdict = "MIXED: report both numbers, claim neither cleanly."
    log(f">>> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"means": means, "fvu_train": ftrain, "fvu_probe": fprobe, "ratio": ratio,
         "verdict": verdict, "per_image": out}, indent=2))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
