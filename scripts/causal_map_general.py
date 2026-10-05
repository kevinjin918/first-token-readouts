"""Does the readout/control story generalise beyond cardiomegaly?

Everything causal in this project is cardiomegaly-only, which is the most obvious objection to the
whole line. This builds the causal map for an arbitrary finding under the **normalised** readout,
so the second finding is measured correctly from the start rather than audited afterwards.

It produces, in one pass over 64 occlusion blocks per film, the three things the cardiomegaly line
needed four scripts and a retrofit to get:

  CONCENTRATION (the A2 analog)   share of total causal effect in the top 5 of 64 blocks. Uniform
                                  would be 5/64 = 7.8%. Cardiomegaly gives 72%.
  ATTENTION-VS-CAUSE (A4/A5)      within-film Spearman rho between a block's attention weight and
                                  its own causal effect, against a shuffled-attention control.
                                  Cardiomegaly gives rho = -0.067 vs -0.009 shuffled.
  THE MAP ITSELF                  per-block effects, which is the input the A6 feature probe needs.

Every probability is recorded three ways -- raw single-token, normalised over the yes/no pair, and
answer mass -- because `readout_audit.md` showed the raw readout inflated a clamp effect 150x. The
reported `block_effect` is the **normalised** one; `block_effect_raw` is kept for comparison with
the banked cardiomegaly map, which was built raw.

FILM SELECTION. `--require-label positive` takes films the dataset labels with the finding AND the
model calls positive: genuine detection. This differs from the banked cardiomegaly map, which took
label-NEGATIVE AP films where the model fired, i.e. false positives. Both are legitimate; they are
different questions, and mixing them silently would be the mistake. Default is `positive`.

DECISION RULE. The first version of this gate was genuinely pre-specified but MIS-SPECIFIED: it
tested the top-5 SHARE of causal effect, a ratio with no magnitude floor, and certified effusion as
LOCALISED on a mean top-5 block effect of +0.036. The gate below was tightened AFTER seeing that
result, by adding an absolute floor on the single best block and a per-film count, and a third
verdict (HETEROGENEOUS) was added. We state this rather than presenting the revised gate as
pre-registered. Runs made before the revision keep their original verdict in the result file
alongside the corrected one. The rule now in force:
  LOCALISED      top-5 share >= 0.40 (vs 0.078 uniform) AND the mean border-block effect is below
                 a quarter of the mean top-5 effect. The finding has a spatial causal locus, so the
                 A6 probe is worth running on it.
  DIFFUSE        top-5 share < 0.20. Evidence is spread; report that as the finding's character and
                 do NOT run the A6 probe, which presupposes separable causal and inert blocks.
  in between     report the number and claim neither.

GPU. Usage (VM):
  python scripts/causal_map_general.py --finding effusion --n-films 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

GRID, BLOCK = 16, 2
NB = GRID // BLOCK
FIRE = 0.5
TOP_K = 5
PROMPTS = {
    "cardiomegaly": "Does this chest X-ray show cardiomegaly? Answer yes or no.",
    "effusion": "Does this chest X-ray show a pleural effusion? Answer yes or no.",
    "pneumothorax": "Does this chest X-ray show a pneumothorax? Answer yes or no.",
    "nodule": "Does this chest X-ray show a pulmonary nodule? Answer yes or no.",
    "mass": "Does this chest X-ray show a mass? Answer yes or no.",
    "atelectasis": "Does this chest X-ray show atelectasis? Answer yes or no.",
}
LABELS = {"cardiomegaly": "Cardiomegaly", "effusion": "Effusion",
          "pneumothorax": "Pneumothorax", "nodule": "Nodule", "mass": "Mass",
          "atelectasis": "Atelectasis"}


def log(*a: object) -> None:
    print(*a, flush=True)


def occlude(arr, br, bc):
    a = arr.copy()
    h, w = a.shape[:2]
    r0, r1 = int(br * BLOCK * h / GRID), int((br + 1) * BLOCK * h / GRID)
    c0, c1 = int(bc * BLOCK * w / GRID), int((bc + 1) * BLOCK * w / GRID)
    a[r0:r1, c0:c1] = 0
    return Image.fromarray(a)


def main() -> None:
    import torch  # noqa: PLC0415
    from scipy.stats import spearmanr  # noqa: PLC0415
    from tracecxr.core.config import MODELS  # noqa: PLC0415
    from tracecxr.models._base import resolve_yes_no_token_ids, yes_probability  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finding", default="effusion", choices=sorted(PROMPTS))
    ap.add_argument("--require-label", default="positive",
                    choices=("positive", "negative", "any"))
    ap.add_argument("--view", default="any", choices=("any", "AP", "PA"),
                    help="the banked cardiomegaly map was AP-only, which is where the view "
                         "shortcut lives; isolating it needs this held fixed")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n-films", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"/mnt/fast/clt/causal_map_{args.finding}.json")

    locator = MODELS["medgemma"].locator
    proc = AutoProcessor.from_pretrained(locator)
    model = AutoModelForImageTextToText.from_pretrained(
        locator, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tok = getattr(proc, "tokenizer", proc)
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    img_tok = int(model.config.image_token_index)
    prompt = PROMPTS[args.finding]
    log(f"loaded MedGemma | finding={args.finding} | prompt={prompt!r}")

    def run(image, want_attn=False):
        inp = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to("cuda")
        d = inp["input_ids"].shape[1] - 1
        with torch.no_grad():
            o = model(**inp, output_attentions=want_attn)
        lg = o.logits[0, d].float()
        pv = torch.softmax(lg, -1)
        raw = float(pv[yes_ids].max())
        nrm = float(yes_probability(lg, yes_ids, no_ids))
        mass = float(pv[yes_ids].max() + pv[no_ids].max())
        att = None
        if want_attn:
            ids = inp["input_ids"][0]
            cols = (ids == img_tok).nonzero(as_tuple=True)[0]
            row = torch.stack([a[0, :, d, :].float().mean(0) for a in o.attentions]).mean(0)
            att = row[cols].cpu().numpy().reshape(GRID, GRID)
        return raw, nrm, mass, att

    import pandas as pd  # noqa: PLC0415
    df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
    img_dir = args.data_dir / "images"
    lab = LABELS[args.finding]
    pool = []
    for _, r in df.iterrows():
        name = str(r["Image Index"])
        if name.startswith("._") or not (img_dir / name).exists():
            continue
        has = lab in str(r["Finding Labels"])
        if args.require_label == "positive" and not has:
            continue
        if args.require_label == "negative" and has:
            continue
        if args.view != "any" and str(r["View Position"]) != args.view:
            continue
        pool.append(name)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    log(f"{len(pool)} candidate films (require-label={args.require_label}, view={args.view})")

    recs, scanned = [], 0
    for i in order:
        if len(recs) >= args.n_films:
            break
        name = pool[int(i)]
        scanned += 1
        arr = np.array(Image.open(img_dir / name).convert("RGB"))
        raw, nrm, mass, att = run(Image.open(img_dir / name).convert("RGB"), want_attn=True)
        if raw < FIRE:
            continue
        b_att, b_eff, b_eff_raw, b_mass = [], [], [], []
        for br in range(NB):
            for bc in range(NB):
                b_att.append(float(att[br * BLOCK:(br + 1) * BLOCK,
                                       bc * BLOCK:(bc + 1) * BLOCK].sum()))
                r2, n2, m2, _ = run(occlude(arr, br, bc))
                b_eff.append(nrm - n2)
                b_eff_raw.append(raw - r2)
                b_mass.append(m2)
        rho, p = spearmanr(b_att, b_eff)
        rho_sh, _ = spearmanr(rng.permutation(b_att), b_eff)
        eff = np.array(b_eff)
        top = np.argsort(-eff)[:TOP_K]
        share = float(eff[top].sum() / eff.sum()) if eff.sum() > 1e-9 else float("nan")
        recs.append({"image": name, "base": nrm, "base_raw": raw, "base_mass": mass,
                     "rho": float(rho), "p": float(p), "rho_shuffled": float(rho_sh),
                     "top5_share": share, "block_attn": b_att,
                     "block_effect": b_eff, "block_effect_raw": b_eff_raw,
                     "block_mass": b_mass})
        log(f"  {name}: base norm {nrm:.3f} (raw {raw:.3f})  top5 share {share:.3f}  "
            f"rho {rho:+.3f} (shuf {rho_sh:+.3f})")

    if not recs:
        raise SystemExit(f"no firing films for {args.finding} in {scanned} scanned")

    idx = np.arange(NB * NB)
    R, C = idx // NB, idx % NB
    border = (R == 0) | (R == NB - 1) | (C == 0) | (C == NB - 1)
    eff_all = np.array([r["block_effect"] for r in recs])
    shares = np.array([r["top5_share"] for r in recs])
    top_mean = float(np.mean([np.sort(e)[-TOP_K:].mean() for e in eff_all]))
    bor_mean = float(eff_all[:, border].mean())
    rhos = np.array([r["rho"] for r in recs])
    rhos_sh = np.array([r["rho_shuffled"] for r in recs])

    log(f"\n=== {args.finding}: n={len(recs)} firing films ({scanned} scanned) ===")
    log(f"  baseline norm {np.mean([r['base'] for r in recs]):.3f}, "
        f"raw {np.mean([r['base_raw'] for r in recs]):.3f}, "
        f"mass {np.mean([r['base_mass'] for r in recs]):.3f}")
    log(f"  top-{TOP_K} share of total effect  {np.nanmean(shares):.3f}   "
        f"(uniform {TOP_K / 64:.3f})")
    log(f"  mean top-{TOP_K} block effect       {top_mean:+.3f}")
    log(f"  mean border block effect         {bor_mean:+.3f}   "
        f"ratio {bor_mean / top_mean if top_mean else float('nan'):.3f}")
    log(f"  attention vs cause, mean rho     {rhos.mean():+.3f} "
        f"(shuffled {rhos_sh.mean():+.3f}, {int((rhos > 0).sum())}/{len(rhos)} positive)")

    # A share is a ratio and says nothing about size: effusion returned share 1.09 while its mean
    # top-5 block effect was +0.036, i.e. concentration of a negligible total. The gate therefore
    # needs an absolute magnitude floor on the single best block, and the per-film count matters
    # more than the mean because findings can be removable on some films and not others.
    best = eff_all.max(axis=1)
    removable = int((best > 0.15).sum())
    share_m = float(np.nanmean(shares))
    log(f"  single BEST block effect         {best.mean():+.3f} "
        f"(max {best.max():+.3f}); films with a block worth >0.15: {removable}/{len(recs)}")

    localised = (share_m >= 0.40 and bor_mean < 0.25 * top_mean
                 and best.mean() >= 0.15 and removable >= 0.6 * len(recs))
    diffuse = best.mean() < 0.05
    verdict = ("LOCALISED: a removable spatial causal locus exists on most films; "
               "the A6 feature probe is worth running" if localised else
               "DIFFUSE: no single block removes the evidence; do not run the A6 probe"
               if diffuse else
               f"HETEROGENEOUS: a removable locus on {removable}/{len(recs)} films only "
               f"(mean best block {best.mean():+.3f}). Report per-film, not as a group claim")
    log(f">>> {verdict}")

    out.write_text(json.dumps(
        {"finding": args.finding, "require_label": args.require_label, "n": len(recs),
         "n_scanned": scanned, "readout": "normalised (block_effect); raw kept alongside",
         "top5_share": share_m, "top5_effect": top_mean, "border_effect": bor_mean,
         "best_block_mean": float(best.mean()), "best_block_max": float(best.max()),
         "n_removable": removable,
         "rho": float(rhos.mean()), "rho_shuffled": float(rhos_sh.mean()),
         "verdict": verdict, "records": recs}, indent=2))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
