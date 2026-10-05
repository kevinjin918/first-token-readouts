"""Is the occlusion ground truth a decision effect, or the same answer-mass artifact?

`readout_audit.json` established that CLAMPING never moved the yes/no decision: on the exact A13/A16
films the raw readout shows a 0.294 drop while the normalised readout shows **+0.002**, because the
clamp pushes ~30 points of probability mass off the answer tokens entirely (mass 0.921 -> 0.628).
The raw readout scored that mass shift as suppression.

`evidence_localization.py` uses the SAME raw single-token readout (`softmax(logits)[yes_id]`, line
155). So the occlusion result -- top block +0.700, 72% of the effect in 5 of 64 blocks, five
pre-specified controls -- has never been checked against this confound. It is the causal ground
truth that A2, A3, A6 and the paper's whole wedge rest on. If it is also a mass artifact, the
project has no causal anchor; if it survives, the contrast between occlusion (a real input
intervention) and clamping (a representation intervention) becomes the sharpest result here.

This replicates the original occlusion exactly -- same `fill()` geometry, same three fill types,
same border-block negative control, same banked films and their recorded top block -- and reads
BOTH probabilities plus the answer mass from one forward pass.

PRE-SPECIFIED, fixed before running. "Drop" is baseline minus occluded, on the target finding.
  GROUND TRUTH SURVIVES   normalised drop at the top block > 0.15 AND > 3x the normalised drop at
                          the border control. Occlusion moves the decision; the A2/A3/A6 line
                          stands and the clamp/occlusion contrast is real.
  MASS ARTIFACT           normalised drop < 0.05 while the raw drop exceeds 0.30. Occlusion also
                          only moves answer-format propensity, and A2/A3 must be restated.
  PARTIAL                 anything between; report both columns and claim only what the normalised
                          column supports.

GPU. Usage (VM):
  python scripts/occlusion_readout_audit.py --banked attn_causal_map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

GRID, BLOCK = 16, 2
NB = GRID // BLOCK
PROMPTS = {
    "cardiomegaly": "Does this chest X-ray show cardiomegaly? Answer yes or no.",
    "effusion": "Does this chest X-ray show a pleural effusion? Answer yes or no.",
    "pneumothorax": "Does this chest X-ray show a pneumothorax? Answer yes or no.",
}
TARGET = "cardiomegaly"
FILLS = ("black", "mean", "blur")


def log(*a: object) -> None:
    print(*a, flush=True)


def block_slice(a, b):
    h, w = a.shape[:2]
    br, bc = divmod(int(b), NB)
    r0, r1 = br * BLOCK, (br + 1) * BLOCK
    c0, c1 = bc * BLOCK, (bc + 1) * BLOCK
    return (slice(int(r0 * h / GRID), int(r1 * h / GRID)),
            slice(int(c0 * w / GRID), int(c1 * w / GRID)))


def fill(arr, b, how):
    a = arr.copy()
    ys, xs = block_slice(a, b)
    if how == "black":
        a[ys, xs] = 0
    elif how == "mean":
        a[ys, xs] = int(arr.mean())
    elif how == "blur":
        patch = Image.fromarray(arr[ys, xs])
        a[ys, xs] = np.array(patch.filter(ImageFilter.GaussianBlur(radius=12)))
    return Image.fromarray(a)


def main() -> None:
    import torch  # noqa: PLC0415
    from tracecxr.core.config import MODELS  # noqa: PLC0415
    from tracecxr.models._base import resolve_yes_no_token_ids, yes_probability  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--banked", type=Path, default=Path("/mnt/fast/clt/attn_causal_map.json"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n-films", type=int, default=20)
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--out", default="/mnt/fast/clt/occlusion_readout_audit.json")
    args = ap.parse_args()

    mid = MODELS["medgemma"].locator
    proc = AutoProcessor.from_pretrained(mid)
    # .to("cuda") rather than device_map=: `accelerate` is not installed in vox_venv.
    model = AutoModelForImageTextToText.from_pretrained(
        mid, torch_dtype=torch.bfloat16).to("cuda").eval()
    tok = getattr(proc, "tokenizer", proc)
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    log(f"loaded MedGemma; {len(yes_ids)} yes ids, {len(no_ids)} no ids")

    def run(image, prompt):
        inp = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to("cuda")
        d = inp["input_ids"].shape[1] - 1
        with torch.no_grad():
            lg = model(**inp).logits[0, d].float()
        pv = torch.softmax(lg, -1)
        raw = float(pv[yes_ids].max())
        mass = float(pv[yes_ids].max() + pv[no_ids].max())
        return raw, float(yes_probability(lg, yes_ids, no_ids)), mass

    banked = json.loads(args.banked.read_text())["A"]["records"][: args.n_films]
    img_dir = args.data_dir / "images"
    rng = np.random.default_rng(args.seed)
    idx = np.arange(NB * NB)
    R, C = idx // NB, idx % NB
    border_blocks = idx[(R == 0) | (R == NB - 1) | (C == 0) | (C == NB - 1)]

    rows: list[dict] = []
    for b in banked:
        path = img_dir / b["image"]
        if not path.exists():
            continue
        arr = np.array(Image.open(path).convert("RGB"))
        top = int(np.argmax(b["block_effect"]))
        rnd_border = int(rng.choice(border_blocks))
        raw, nrm, mass = run(Image.open(path).convert("RGB"), PROMPTS[TARGET])
        if raw < 0.5:
            continue
        rec = {"image": b["image"], "top_block": top,
               "base": {"raw": raw, "norm": nrm, "mass": mass}}
        for how in FILLS:
            r, n, m = run(fill(arr, top, how), PROMPTS[TARGET])
            rec[how] = {"raw": r, "norm": n, "mass": m}
        r, n, m = run(fill(arr, rnd_border, "black"), PROMPTS[TARGET])
        rec["border"] = {"raw": r, "norm": n, "mass": m}
        for f in PROMPTS:
            if f == TARGET:
                continue
            r0, n0, m0 = run(Image.open(path).convert("RGB"), PROMPTS[f])
            r1, n1, m1 = run(fill(arr, top, "black"), PROMPTS[f])
            rec[f"{f}_base"] = {"raw": r0, "norm": n0, "mass": m0}
            rec[f"{f}_occ"] = {"raw": r1, "norm": n1, "mass": m1}
        rows.append(rec)
        log(f"  {rec['image']}: base raw {raw:.3f} norm {nrm:.3f}  ->  black raw "
            f"{rec['black']['raw']:.3f} norm {rec['black']['norm']:.3f} "
            f"mass {rec['black']['mass']:.3f}")

    if not rows:
        raise SystemExit("no firing films")

    log(f"\n=== n={len(rows)} firing films, occlusion at the recorded top-effect block ===")
    log(f"  baseline: raw {np.mean([r['base']['raw'] for r in rows]):.3f}  "
        f"norm {np.mean([r['base']['norm'] for r in rows]):.3f}  "
        f"mass {np.mean([r['base']['mass'] for r in rows]):.3f}")
    log(f"\n  {'arm':22s}{'raw drop':>10s}{'NORM drop':>11s}{'mass after':>12s}")
    out = {}
    for arm in (*FILLS, "border"):
        d_raw = float(np.mean([r["base"]["raw"] - r[arm]["raw"] for r in rows]))
        d_nrm = float(np.mean([r["base"]["norm"] - r[arm]["norm"] for r in rows]))
        mass = float(np.mean([r[arm]["mass"] for r in rows]))
        out[arm] = {"raw_drop": d_raw, "norm_drop": d_nrm, "mass_after": mass}
        label = arm + ("-fill" if arm in FILLS else " (neg control)")
        log(f"  {label:22s}{d_raw:>10.3f}{d_nrm:>11.3f}{mass:>12.3f}")

    log("\n  other findings under the SAME occlusion (should be ~0 if evidence is specific):")
    for f in PROMPTS:
        if f == TARGET:
            continue
        fires = np.array([r[f"{f}_base"]["raw"] for r in rows]) >= 0.5
        d_raw = np.array([r[f"{f}_base"]["raw"] - r[f"{f}_occ"]["raw"] for r in rows])
        d_nrm = np.array([r[f"{f}_base"]["norm"] - r[f"{f}_occ"]["norm"] for r in rows])
        out[f] = {"n_firing": int(fires.sum()),
                  "raw_drop": float(d_raw[fires].mean()) if fires.sum() else float("nan"),
                  "norm_drop": float(d_nrm[fires].mean()) if fires.sum() else float("nan")}
        log(f"  {f:22s}{out[f]['raw_drop']:>10.3f}{out[f]['norm_drop']:>11.3f}"
            f"   (n firing {int(fires.sum())})")

    top_n, bor_n = out["black"]["norm_drop"], out["border"]["norm_drop"]
    survives = top_n > 0.15 and top_n > 3 * max(bor_n, 0.0)
    artifact = top_n < 0.05 and out["black"]["raw_drop"] > 0.30
    verdict = ("GROUND TRUTH SURVIVES: occlusion moves the decision, not just the answer mass"
               if survives else
               "MASS ARTIFACT: occlusion only moves answer-format propensity" if artifact else
               "PARTIAL: claim only what the normalised column supports")
    log(f"\n>>> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"n": len(rows), "arms": out, "verdict": verdict, "rows": rows}, indent=2))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
