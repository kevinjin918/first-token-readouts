"""Is the donor-set overlap a property of the selection rule, or of the patients?

The paper's mechanism claim is that top-activation selection is finding-agnostic: feature sets built
from cardiomegaly-, effusion- and pneumothorax-positive films overlap on 77 of 80 features (mean
pairwise Jaccard 0.928 at N=80), so an operator applied to "the cardiomegaly features" has nothing
finding-specific to act on. A reviewer pointed out the flaw in that evidence. The donor films were
the first eight firing films in patient-ordered CSV order, so the cardiomegaly set came from 3
patients, effusion from 5, and pneumothorax from one (00000013), who also appears in the other two
sets and among the test films. Overlap could be patient identity, not the selection rule.

This rebuilds every set so that patient identity cannot contribute:
  - one film per patient, and no patient in more than one set (donor or negative);
  - no donor or test patient shared with the 20 banked readout films;
  - films drawn in a seeded random order, the three findings taking turns so none gets first pick;
  - a donor must carry the NIH label AND the model must call it (P_norm >= 0.5); a contrastive
    negative must lack the label and the model must not call it (P_norm < 0.5).
Then it repeats the two measurements, recording P_raw, P_norm, answer mass and the yes-minus-no
logit gap for every amplification (the original run recorded P_raw only).

  JACCARD        mean pairwise Jaccard between the three findings' sets, top-activation and
                 contrastive selection, at N = 20 / 80 / 320
  AMPLIFICATION  on 12 cardiomegaly-negative test films in the original band (0.05 < P_raw < 0.5),
                 rise in P(cardiomegaly) when amplifying each N=80 set x3 and x10: card_donor,
                 eff_donor, ptx_donor, self, random

PRE-SPECIFIED, fixed before running.
  Jaccard, activation selection, N=80:
    STILL FINDING-AGNOSTIC   > 0.70. Patient identity was not the cause; the paper's claim stands.
    PATIENT-DRIVEN           < 0.50. The overlap was mostly shared patients; the claim is withdrawn.
    INTERMEDIATE             otherwise; report the number and weaken the claim to match.
  Amplification at x10, on P_norm (the paper's decision readout), same thresholds as the original:
    DONOR-SPECIFIC   card_donor exceeds both foreign donors by > 0.10 mean rise, Wilcoxon p < 0.05.
    GENERIC          card_donor within 0.10 of both foreign donors, and > 0.10 above random.
    NO EFFECT        card_donor <= 0.10 above random.
    MIXED            anything else.
  The same rule is also evaluated on P_raw for continuity with the original, and reported.

GPU. Usage (VM):
  python scripts/donor_sets_patient_disjoint.py
"""

from __future__ import annotations

import argparse
import itertools
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
FINDINGS = tuple(PROMPTS)
TARGET = "cardiomegaly"
COUNTS = (20, 80, 320)
N_FEATS = 80
AMPS = (3.0, 10.0)
BAND = (0.05, 0.5)
SETS = ("card_donor", "eff_donor", "ptx_donor", "self", "random")


def log(*a: object) -> None:
    print(*a, flush=True)


def patient(name: str) -> int:
    """NIH image names are ``<8-digit patient id>_<3-digit follow-up>.png``."""
    return int(name.split("_")[0])


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def amp_verdict(rise: dict, p: dict) -> str:
    """The original donor-specificity rule, on a ``{set: mean rise}`` dict at the top amplitude."""
    gaps = {f: rise["card_donor"] - rise[f] for f in ("eff_donor", "ptx_donor")}
    over_random = rise["card_donor"] - rise["random"]
    if all(gaps[f] > 0.10 and p[f] < 0.05 for f in gaps):
        return "DONOR-SPECIFIC"
    if over_random <= 0.10:
        return "NO EFFECT"
    if all(abs(g) <= 0.10 for g in gaps.values()):
        return "GENERIC"
    return "MIXED"


def main() -> None:  # noqa: PLR0915
    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from scipy.stats import wilcoxon  # noqa: PLC0415
    from tracecxr.core.config import MODELS  # noqa: PLC0415
    from tracecxr.intervention import ClampedMedGemma, FeatureEdit  # noqa: PLC0415
    from tracecxr.models._base import resolve_yes_no_token_ids, yes_probability  # noqa: PLC0415
    from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/clt_scale"))
    ap.add_argument("--ckpt", default="/mnt/fast/clt/orig2048/clt_ckpt.pt")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n-donor", type=int, default=8)
    ap.add_argument("--n-films", type=int, default=12)
    ap.add_argument("--max-scan", type=int, default=400,
                    help="films tried per set before giving up")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/fast/clt/donor_sets_patient_disjoint.json"))
    args = ap.parse_args()

    res = json.loads((args.results / "clt_live_result.json").read_text())
    layers = list(res["layers"])
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                    k=res.get("k", 32), n_layers=len(layers),
                    adam_8bit=bool(res.get("adam_8bit", False)))
    cmg = ClampedMedGemma(args.ckpt, layers, clt_config=cfg, model_id=MODELS["medgemma"].locator)
    cmg._load()
    st = cmg._state
    proc, model, clt = st["proc"], st["model"], st["clt"]
    tok = getattr(proc, "tokenizer", proc)
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    img_tok = int(model.config.image_token_index)
    rng = np.random.default_rng(args.seed)
    log(f"loaded MedGemma + CLT ({cfg.n_features} feats x {len(layers)} layers)")

    def readouts(image, finding, pairs=(), amt=1.0):
        edits = [FeatureEdit(layer=L, feature=f, op="scale", amount=amt, positions="image")
                 for L, f in pairs]
        row = cmg.decision_logits_row(prompt=PROMPTS[finding], image=image, edits=edits).float()
        pv = torch.softmax(row, -1)
        y, n = pv[yes_ids].max(), pv[no_ids].max()
        return {"raw": float(y), "norm": float(yes_probability(row, yes_ids, no_ids)),
                "mass": float(y + n), "gap": float(row[yes_ids].max() - row[no_ids].max())}

    def mean_feats(image, prompt):
        """Identical to `clamp_amplify_donor_specificity.mean_feats`."""
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
        cols = (inputs["input_ids"][0] == img_tok).nonzero(as_tuple=True)[0]
        mlp_in = torch.stack([cap[L].float() for L in layers], dim=1)
        with torch.no_grad():
            f = clt.encode(mlp_in)
        return f[cols].mean(0).cpu().numpy()

    def topn(score, n):
        flat = np.argsort(-score, axis=None)[:n]
        return [(layers[int(i // score.shape[1])], int(i % score.shape[1])) for i in flat]

    df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
    img_dir = args.data_dir / "images"
    have = {p.name for p in img_dir.glob("*.png")}
    df = df[df["Image Index"].isin(have)]
    films = [(str(r["Image Index"]), str(r["Finding Labels"])) for _, r in df.iterrows()]
    order = rng.permutation(len(films))
    films = [films[i] for i in order]
    banked = json.loads((args.results / "readout_audit_banked.json").read_text())["fire_rows"]
    used = {patient(r["image"]) for r in banked}      # never reuse a readout-film patient
    log(f"{len(films)} local films; {len(used)} banked patients excluded")

    def draw(finding, positive, out, cursor):
        """Advance ``cursor`` to the next eligible film for this set; return it or None."""
        while cursor[0] < len(films) and cursor[1] < args.max_scan:
            name, lab = films[cursor[0]]
            cursor[0] += 1
            if patient(name) in used or (LABELS[finding] in lab) != positive:
                continue
            cursor[1] += 1
            img = Image.open(img_dir / name).convert("RGB")
            r = readouts(img, finding)
            if (r["norm"] >= 0.5) != positive:
                continue                               # the model must agree with the label
            used.add(patient(name))
            out.append({"image": name, **r, "feats": mean_feats(img, PROMPTS[finding])})
            return out[-1]
        return None

    # findings take turns, so a patient carrying two labels goes to whichever set reaches it first
    picks: dict[tuple[str, bool], list] = {(f, s): [] for f in FINDINGS for s in (True, False)}
    cursors = {k: [0, 0] for k in picks}
    for positive in (True, False):
        live = list(FINDINGS)
        while live:
            for f in list(live):
                k = (f, positive)
                if len(picks[k]) >= args.n_donor or draw(f, positive, picks[k], cursors[k]) is None:
                    live.remove(f)
        for f in FINDINGS:
            log(f"  {f:13s} {'pos' if positive else 'neg'}: {len(picks[(f, positive)])} films, "
                f"{len({patient(x['image']) for x in picks[(f, positive)]})} patients")
    for f in FINDINGS:
        if len(picks[(f, True)]) < args.n_donor or len(picks[(f, False)]) < args.n_donor:
            raise SystemExit(f"could not fill patient-disjoint sets for {f}")
    all_pat = [patient(x["image"]) for v in picks.values() for x in v]
    assert len(all_pat) == len(set(all_pat)), "a patient appears twice"

    scores = {}
    for f in FINDINGS:
        mp = np.mean([x["feats"] for x in picks[(f, True)]], axis=0)
        mn = np.mean([x["feats"] for x in picks[(f, False)]], axis=0)
        scores[f] = {"activation": mp, "contrast": mp - mn}

    log("\n=== mean pairwise Jaccard, patient-disjoint sets ===")
    jac: dict[str, dict[int, float]] = {}
    for crit in ("activation", "contrast"):
        jac[crit] = {}
        for n in COUNTS:
            sets_n = {f: topn(scores[f][crit], n) for f in FINDINGS}
            jac[crit][n] = float(np.mean([jaccard(sets_n[a], sets_n[b])
                                          for a, b in itertools.combinations(FINDINGS, 2)]))
        log(f"  {crit:11s}" + "".join(f"  N={n}: {jac[crit][n]:.3f}" for n in COUNTS))
    donors = {f: topn(scores[f]["activation"], N_FEATS) for f in FINDINGS}
    overlap80 = {f"{a}x{b}": len(set(donors[a]) & set(donors[b]))
                 for a, b in itertools.combinations(FINDINGS, 2)}
    log(f"  activation overlap at N=80: {overlap80}")
    j80 = jac["activation"][N_FEATS]
    jac_verdict = ("STILL FINDING-AGNOSTIC" if j80 > 0.70 else
                   "PATIENT-DRIVEN" if j80 < 0.50 else "INTERMEDIATE")
    log(f">>> Jaccard: {jac_verdict} ({j80:.3f})")

    # --- amplification on band films, patient-disjoint from every donor and readout film ---
    rows: list[dict] = []
    for name, lab in films:
        if len(rows) >= args.n_films:
            break
        if LABELS[TARGET] in lab or patient(name) in used:
            continue
        img = Image.open(img_dir / name).convert("RGB")
        base = readouts(img, TARGET)
        if not (BAND[0] < base["raw"] < BAND[1]):
            continue
        used.add(patient(name))
        sets = {"card_donor": donors["cardiomegaly"], "eff_donor": donors["effusion"],
                "ptx_donor": donors["pneumothorax"],
                "self": topn(mean_feats(img, PROMPTS[TARGET]), N_FEATS),
                "random": [(layers[int(rng.integers(len(layers)))],
                            int(rng.integers(cfg.n_features))) for _ in range(N_FEATS)]}
        rec = {"image": name, "base": base}
        for s, pairs in sets.items():
            for amt in AMPS:
                rec[f"{s}_{amt:g}"] = readouts(img, TARGET, pairs, amt)
        rows.append(rec)
        log(f"  {name}: base raw {base['raw']:.3f} norm {base['norm']:.3f}  x10 norm " +
            "  ".join(f"{s.split('_')[0]} {rec[f'{s}_10']['norm']:.2f}" for s in SETS))
    if len(rows) < 6:
        raise SystemExit(f"only {len(rows)} band films")

    amp: dict[str, dict] = {}
    for readout in ("norm", "raw", "mass", "gap"):
        rise = {}
        for s in SETS:
            for a in AMPS:
                d = np.array([r[f"{s}_{a:g}"][readout] - r["base"][readout] for r in rows])
                rise[f"{s}_{a:g}"] = d
        top = {s: float(rise[f"{s}_10"].mean()) for s in SETS}
        p = {f: float(wilcoxon(rise["card_donor_10"], rise[f"{f}_10"]).pvalue)
             for f in ("eff_donor", "ptx_donor")}
        amp[readout] = {"rise": {k: float(v.mean()) for k, v in rise.items()}, "p": p,
                        "verdict": amp_verdict(top, p) if readout in ("norm", "raw") else None}
        log(f"  {readout:5s} rise x10: " + "  ".join(f"{s} {top[s]:+.3f}" for s in SETS) +
            (f"   -> {amp[readout]['verdict']}" if amp[readout]["verdict"] else ""))

    log(f">>> amplification (P_norm): {amp['norm']['verdict']}   (P_raw: {amp['raw']['verdict']})")
    for v in picks.values():
        for x in v:
            x.pop("feats")
    args.out.write_text(json.dumps(
        {"seed": args.seed, "n_donor": args.n_donor, "n_films": len(rows),
         "donor_films": {f"{f}_{'pos' if s else 'neg'}": v for (f, s), v in picks.items()},
         "jaccard": jac, "overlap80": overlap80, "jaccard_verdict": jac_verdict,
         "amplification": amp, "rows": rows}, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
