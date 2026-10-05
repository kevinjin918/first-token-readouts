"""Do the CLT features distinguish causally-important image patches from inert ones?

Motivation. `clamp_causal_anchored.py` found that clamping features anchored to occlusion-verified
causal blocks drops P(cardiomegaly) by 0.338, against 0.004 for random features -- but ALSO 0.201
for features anchored to blocks with near-zero measured causal effect. Causal beats inert by only
+0.138 (Wilcoxon p=0.0425, 10/12 films). So anchoring works, but its specificity to the causal
region is partial.

This asks why, at the representational level rather than the intervention level: **can the feature
basis tell a causal patch from an inert one at all?**

  If YES (patches separable from features) -> the partial specificity of the clamp is a property of
      the intervention, not of the representation; the information is there to select on better.
  If NO (features do not separate them)    -> the representation does not encode evidence-bearing
      regions distinctly, which directly explains why anchoring gives only partial specificity, and
      bounds what ANY feature-selection method can achieve on this dictionary.

PRE-SPECIFIED, fixed before running. Leave-one-film-out logistic probe on mean feature activations,
causal blocks vs inert blocks, pooled across films:
  AUC >= 0.75  -> features separate them clearly.
  AUC <= 0.60  -> features do not separate them.
  in between   -> weak separation; report the number, claim neither.

CONTROLS
  (a) label-shuffled probe, **20 seeds**: the null band must contain 0.5 and must not contain the
      observed AUC. A single shuffle is not a null distribution -- the first version of this script
      drew one and got 0.372, ~3 SE below chance, which was uninterpretable on its own.
  (b) raw-pixel baseline: mean intensity + sd of the patch region. If pixels separate them as well
      as features do, the features add nothing over trivial image statistics.

Validity: everything here is measured at IMAGE token positions, whose activations are
prompt-independent under causal masking (image FVU 0.355, identical across five prompts).

GPU. Usage (VM):
  python scripts/feats_causal_vs_inert.py --ckpt <clt> --result <result.json> \
    --banked attn_causal_map.json --n-films 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

GRID, BLOCK = 16, 2
NB = GRID // BLOCK
PROMPT = "Does this chest X-ray show cardiomegaly? Answer yes or no."
TOP_BLOCKS = 5


def log(*a: object) -> None:
    print(*a, flush=True)


def block_patch_ids(block: int) -> list[int]:
    br, bc = divmod(int(block), NB)
    return [r * GRID + c
            for r in range(br * BLOCK, (br + 1) * BLOCK)
            for c in range(bc * BLOCK, (bc + 1) * BLOCK)]


def block_pixels(arr: np.ndarray, block: int) -> np.ndarray:
    h, w = arr.shape[:2]
    br, bc = divmod(int(block), NB)
    return arr[int(br * BLOCK * h / GRID):int((br + 1) * BLOCK * h / GRID),
               int(bc * BLOCK * w / GRID):int((bc + 1) * BLOCK * w / GRID)]


def main() -> None:
    import torch  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from tracecxr.attribution.medgemma_backend import MedGemmaCLTBackend  # noqa: PLC0415
    from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/mnt/big/ckpt_live/clt_ckpt.pt")
    ap.add_argument("--result", default=str(Path.home() / "clt_live_result.json"))
    ap.add_argument("--banked", default="/mnt/big/attnpatch/attn_causal_map.json")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n-films", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--n-shuffles", type=int, default=20,
                    help="null draws; one shuffle is not a null distribution")
    ap.add_argument("--out", default="/mnt/big/attnpatch/feats_causal_vs_inert.json")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    layers = list(res["layers"])
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                    k=res.get("k", 32), n_layers=len(layers),
                    adam_8bit=bool(res.get("adam_8bit", False)))
    backend = MedGemmaCLTBackend(args.ckpt, layers, clt_config=cfg)
    backend._load()
    st = backend._state
    proc, model, clt = st["proc"], st["model"], st["clt"]
    img_tok = int(model.config.image_token_index)
    log(f"loaded MedGemma + CLT ({cfg.n_features} feats, span {cfg.span})")

    def feats_per_block(image):
        """Mean CLT feature activation for every block -> (n_blocks, n_layers*n_feat)."""
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": PROMPT}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(model.device)
        cap: dict[int, object] = {}

        def mk(L):
            def hook(_m, inp, _o):  # noqa: ANN001
                cap[L] = inp[0][0]
            return hook

        hs = [st["layers_mod"][L].mlp.register_forward_hook(mk(L)) for L in layers]
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            for h in hs:
                h.remove()
        ids = inputs["input_ids"][0]
        img_cols = (ids == img_tok).nonzero(as_tuple=True)[0]
        mlp_in = torch.stack([cap[L].float() for L in layers], dim=1)
        with torch.no_grad():
            f = clt.encode(mlp_in)
        out = []
        for blk in range(NB * NB):
            sel = img_cols[torch.tensor(block_patch_ids(blk), device=img_cols.device)]
            out.append(f[sel].mean(0).flatten().cpu().numpy())
        return np.stack(out)

    # The original banked map nests records under "A" and its block effects are RAW. Maps from
    # causal_map_general.py are flat and NORMALISED. Accept both so the probe's causal/inert labels
    # can be rebuilt from the corrected readout (readout_audit.md) on the same film population.
    _b = json.loads(Path(args.banked).read_text())
    banked = (_b["A"]["records"] if "A" in _b else _b["records"])[: args.n_films]
    img_dir = args.data_dir / "images"
    X, Xpix, y, film = [], [], [], []

    for fi, b in enumerate(banked):
        path = img_dir / b["image"]
        if not path.exists():
            continue
        eff = np.array(b["block_effect"])
        img = Image.open(path).convert("RGB")
        arr = np.array(img.convert("L"))
        F = feats_per_block(img)
        causal = [int(i) for i in np.argsort(-eff)[:TOP_BLOCKS]]
        inert = [int(i) for i in np.argsort(eff)[:TOP_BLOCKS]]
        for blk, lab in [(c, 1) for c in causal] + [(i, 0) for i in inert]:
            X.append(F[blk])
            px = block_pixels(arr, blk).astype(float)
            Xpix.append([px.mean(), px.std()])
            y.append(lab)
            film.append(fi)
        log(f"  {b['image']}: {len(causal)} causal + {len(inert)} inert blocks")

    X = np.stack(X); Xpix = np.array(Xpix); y = np.array(y); film = np.array(film)
    log(f"\n=== {len(set(film.tolist()))} films, {len(y)} block samples "
        f"({int(y.sum())} causal / {int((1 - y).sum())} inert) ===")

    def loo_auc(features, labels, groups, shuffle=False, seed=0):
        rng = np.random.default_rng(seed)
        lab = rng.permutation(labels) if shuffle else labels
        preds = np.zeros(len(lab), dtype=float)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            if len(set(lab[tr].tolist())) < 2:
                continue
            m = LogisticRegression(max_iter=2000, C=0.1).fit(features[tr], lab[tr])
            preds[te] = m.predict_proba(features[te])[:, 1]
        return float(roc_auc_score(lab, preds))

    auc_feat = loo_auc(X, y, film)
    nulls = np.array([loo_auc(X, y, film, shuffle=True, seed=args.seed + i)
                      for i in range(args.n_shuffles)])
    auc_pix = loo_auc(Xpix, y, film)
    lo, hi = float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))
    p_emp = float((nulls >= auc_feat).mean())

    log(f"  leave-one-film-out AUC, CLT features      {auc_feat:.3f}")
    log(f"  ...shuffled null over {args.n_shuffles} seeds            "
        f"{nulls.mean():.3f} (sd {nulls.std(ddof=1):.3f}, 95% [{lo:.3f}, {hi:.3f}])")
    log(f"  ...empirical p (nulls >= observed)        {p_emp:.4f}")
    log(f"  ...raw pixel mean+sd baseline             {auc_pix:.3f}")
    auc_shuf = float(nulls.mean())

    if auc_feat >= 0.75:
        verdict = ("FEATURES SEPARATE causal from inert patches: the information is "
                   "present, so the "
                   "partial specificity of anchored clamping is a property of the intervention, "
                   "not of the representation.")
    elif auc_feat <= 0.60:
        verdict = ("FEATURES DO NOT SEPARATE them: the representation does not distinguish "
                   "evidence-bearing regions, which explains the partial specificity and bounds "
                   "what any feature-selection method can do on this dictionary.")
    else:
        verdict = "WEAK separation; report the number and claim neither."
    log(f">>> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"n_films": int(len(set(film.tolist()))), "n_samples": int(len(y)),
         "auc_features": auc_feat, "auc_shuffled_mean": auc_shuf,
         "null_sd": float(nulls.std(ddof=1)), "null_ci95": [lo, hi],
         "p_empirical": p_emp, "n_shuffles": int(args.n_shuffles), "auc_pixels": auc_pix,
         "verdict": verdict}, indent=2))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
