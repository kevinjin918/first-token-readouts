"""Is the amplification effect specific to the finding's features, or a generic yes-push?

`clamp_amplify_uncertain.json` reported a 48x selective amplifier: on films where the model is
UNCERTAIN about cardiomegaly (0.05 < P < 0.5), amplifying 80 cardiomegaly-donor features raised
P(cardiomegaly) by 0.172 while effusion rose 0.005 and pneumothorax 0.002.

That number is an artifact of film selection. Films were chosen for cardiomegaly headroom and the
comparison findings then had none: effusion baseline was 0.000 on 8 of 12 films and above 0.9 on the
other 4, pneumothorax 0.000 on 11 of 12. Comparing a finding selected to be contestable against two
findings pinned at floor or ceiling measures the selection, not the model. It is the same headroom
mistake as the `scale +3` ceiling arm in `operator_sweep.md`, in mirror form.

This removes the asymmetry. **One measured quantity** -- the rise in P(cardiomegaly) on the same
uncertain films -- and the only thing that varies is WHICH features are amplified. No finding is
compared against a pinned one, so headroom cannot enter.

FEATURE SETS, all N=80, all at image positions, matched count and matched selection procedure:
  card_donor   top-N mean activation over films the model calls cardiomegaly-positive
  eff_donor    same procedure over effusion-positive films      <- the discriminating control
  ptx_donor    same procedure over pneumothorax-positive films  <- the discriminating control
  self         top-N of the test film itself
  random       N uniformly sampled (layer, feature) pairs       <- machinery control

If a foreign donor set raises P(cardiomegaly) as much as the cardiomegaly donor set does, then
amplification is a generic "say yes" push on the image pathway and the apparent specificity is
nothing. Donor-set overlap is reported alongside, because heavily overlapping sets would make the
comparison vacuous regardless of outcome.

PRE-SPECIFIED, fixed before running, at the strongest operator:
  DONOR-SPECIFIC     card_donor exceeds BOTH foreign donors by > 0.10 mean rise, Wilcoxon p < 0.05
                     against each. Which features you amplify matters -> a real asymmetry against
                     the suppression direction, where it does not.
  GENERIC YES-PUSH   card_donor lands within 0.10 of the foreign donors while all exceed random.
                     Amplification pushes the image pathway toward yes regardless of feature
                     identity -- the up-direction twin of the lesion result.
  NO EFFECT          no set exceeds random by > 0.10.

GPU. Usage (VM):
  python scripts/clamp_amplify_donor_specificity.py --n-films 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROMPTS = {
    "cardiomegaly": "Does this chest X-ray show cardiomegaly? Answer yes or no.",
    "effusion": "Does this chest X-ray show a pleural effusion? Answer yes or no.",
    "pneumothorax": "Does this chest X-ray show a pneumothorax? Answer yes or no.",
}
LABELS = {"cardiomegaly": "Cardiomegaly", "effusion": "Effusion",
          "pneumothorax": "Pneumothorax"}
TARGET = "cardiomegaly"
N_FEATS = 80
AMPS = (3.0, 10.0)
BAND = (0.05, 0.5)      # contestable band for the target finding
SETS = ("card_donor", "eff_donor", "ptx_donor", "self", "random")


def log(*a: object) -> None:
    print(*a, flush=True)


def main() -> None:
    import torch  # noqa: PLC0415
    from tracecxr.core.config import MODELS  # noqa: PLC0415
    from tracecxr.intervention import ClampedMedGemma, FeatureEdit  # noqa: PLC0415
    from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/mnt/fast/clt/orig2048/clt_ckpt.pt")
    ap.add_argument("--result", default="/mnt/fast/clt/orig2048/clt_live_result.json")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n-films", type=int, default=12)
    ap.add_argument("--n-donor", type=int, default=8, help="firing films per donor set")
    ap.add_argument("--donor-scan", type=int, default=400, help="films to scan per donor set")
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--out", default="/mnt/fast/clt/clamp_amplify_donor_specificity.json")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    layers = list(res["layers"])
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                    k=res.get("k", 32), n_layers=len(layers),
                    adam_8bit=bool(res.get("adam_8bit", False)))
    cmg = ClampedMedGemma(args.ckpt, layers, clt_config=cfg,
                          model_id=MODELS["medgemma"].locator)
    cmg._load()
    st = cmg._state
    proc, model, clt = st["proc"], st["model"], st["clt"]
    img_tok = int(model.config.image_token_index)
    rng = np.random.default_rng(args.seed)
    log(f"loaded MedGemma + CLT ({cfg.n_features} feats, span {cfg.span})")

    def mean_feats(image, prompt):
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}],
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
        return f[img_cols].mean(0).cpu().numpy()

    def topn(score, n=N_FEATS):
        flat = np.argsort(-score, axis=None)[:n]
        return [(layers[int(i // score.shape[1])], int(i % score.shape[1])) for i in flat]

    def P(image, finding, pairs=(), amt=0.0):
        edits = [FeatureEdit(layer=L, feature=f, op="scale", amount=amt, positions="image")
                 for L, f in pairs]
        return float(cmg.finding_probability(prompt=PROMPTS[finding], answer="yes",
                                             image=image, edits=edits))

    import pandas as pd  # noqa: PLC0415
    df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
    img_dir = args.data_dir / "images"
    rows_csv = [(str(r["Image Index"]), str(r["Finding Labels"])) for _, r in df.iterrows()]

    # --- donor sets: same procedure per finding, on films the MODEL calls positive ---
    donors: dict[str, list] = {}
    donor_films: dict[str, list[str]] = {}
    for finding in ("cardiomegaly", "effusion", "pneumothorax"):
        acts, used = [], []
        for name, lab in rows_csv[: args.donor_scan]:
            if len(used) >= args.n_donor:
                break
            if LABELS[finding] not in lab or not (img_dir / name).exists():
                continue
            img = Image.open(img_dir / name).convert("RGB")
            if P(img, finding) < 0.5:
                continue
            acts.append(mean_feats(img, PROMPTS[finding]))
            used.append(name)
        if not acts:
            raise SystemExit(f"no firing donor films for {finding}")
        donors[finding] = topn(np.mean(acts, axis=0))
        donor_films[finding] = used
        log(f"  donor set {finding}: {len(used)} films")

    key = {"card_donor": "cardiomegaly", "eff_donor": "effusion", "ptx_donor": "pneumothorax"}
    overlap = {a: {b: len(set(donors[key[a]]) & set(donors[key[b]]))
                   for b in key} for a in key}
    log("  donor-set overlap (of 80): " + ", ".join(
        f"{a}x{b} {overlap[a][b]}" for a in key for b in key if a < b))

    # --- test films: target finding in the contestable band ---
    rows: list[dict] = []
    for name, lab in rows_csv:
        if len(rows) >= args.n_films:
            break
        if LABELS[TARGET] in lab or not (img_dir / name).exists():
            continue
        img = Image.open(img_dir / name).convert("RGB")
        base = P(img, TARGET)
        if not (BAND[0] < base < BAND[1]):
            continue
        sets = {"card_donor": donors["cardiomegaly"], "eff_donor": donors["effusion"],
                "ptx_donor": donors["pneumothorax"],
                "self": topn(mean_feats(img, PROMPTS[TARGET])),
                "random": [(layers[int(rng.integers(len(layers)))],
                            int(rng.integers(cfg.n_features))) for _ in range(N_FEATS)]}
        rec = {"image": name, "base": base}
        for sname, pairs in sets.items():
            for amt in AMPS:
                rec[f"{sname}_{amt:g}"] = P(img, TARGET, pairs, amt)
        rows.append(rec)
        log(f"  {name}: base {base:.3f}  " + "  ".join(
            f"{s.split('_')[0]} {rec[f'{s}_10']:.2f}" for s in SETS))

    if not rows:
        raise SystemExit("no films in the contestable band")

    from scipy.stats import wilcoxon  # noqa: PLC0415

    base = np.array([r["base"] for r in rows])
    log(f"\n=== n={len(rows)} films, P({TARGET}) in {BAND}, median base {np.median(base):.3f} ===")
    log(f"  {'feature set':14s}" + "".join(f"{'x' + str(int(a)):>12s}" for a in AMPS))
    rise = {}
    for sname in SETS:
        line = f"  {sname:14s}"
        for amt in AMPS:
            d = np.array([r[f"{sname}_{amt:g}"] for r in rows]) - base
            rise[(sname, amt)] = d
            line += f"{d.mean():>12.3f}"
        log(line)

    amt = max(AMPS)
    c = rise[("card_donor", amt)]
    gaps, ps = {}, {}
    for foreign in ("eff_donor", "ptx_donor"):
        gaps[foreign] = float(c.mean() - rise[(foreign, amt)].mean())
        ps[foreign] = float(wilcoxon(c, rise[(foreign, amt)]).pvalue)
        log(f"  card_donor - {foreign:10s} = {gaps[foreign]:+.3f}   Wilcoxon p = {ps[foreign]:.4f}")
    over_random = float(c.mean() - rise[("random", amt)].mean())
    log(f"  card_donor - random       = {over_random:+.3f}")

    specific = all(gaps[f] > 0.10 and ps[f] < 0.05 for f in gaps)
    generic = all(abs(gaps[f]) <= 0.10 for f in gaps) and over_random > 0.10
    verdict = ("DONOR-SPECIFIC: which features you amplify matters" if specific else
               "GENERIC YES-PUSH: amplification moves the decision regardless of feature identity"
               if generic else
               "NO EFFECT: nothing exceeds random by > 0.10" if over_random <= 0.10 else
               "MIXED: report the table, claim neither")
    log(f">>> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"n": len(rows), "n_features": N_FEATS, "band": list(BAND), "amps": list(AMPS),
         "donor_films": donor_films, "donor_overlap": overlap,
         "rise": {f"{s}_{a:g}": float(rise[(s, a)].mean()) for s in SETS for a in AMPS},
         "gaps": gaps, "p": ps, "over_random": over_random,
         "verdict": verdict, "rows": rows}, indent=2))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
