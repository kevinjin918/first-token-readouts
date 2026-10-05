"""Does a PROMPT change look like a capability change under the raw readout?

`readout_eval_medgemma.json` is a pre-specified negative: across 200 films and three findings, the
raw and normalised readouts give AUCs within 0.023 and disagree on 2% of yes/no calls. Passive
evaluation is largely safe, because raw ~ normalised x answer-mass and nothing in a fixed-model
sweep systematically moves the mass.

Interventions do move it: clamping drove mass 0.921 -> 0.628 and produced a 150x inflated effect.
That suggests the real scope of the confound is not "interpretability" but **any before/after
comparison on the same model**, where a manipulation can shift how the model formats its answer
while leaving its decision intact. Prompt engineering is the version of that which every applied
group runs, so it is the fair test.

Four phrasings of one question, same films, same model, all three readouts from one pass:
  plain        the format instruction only
  no_format    the same question with the answer-format instruction REMOVED. Expected to move mass
               hard, since nothing now tells the model to open with yes or no.
  persona      a radiologist persona prepended, the most common prompt-engineering move
  gloss        the finding glossed ("cardiomegaly (an enlarged heart)")

PRE-SPECIFIED, fixed before running. Comparing every phrasing against `plain`:
  PROMPT CONFOUND CONFIRMED   some phrasing shifts mean RAW P(yes) by > 0.15 while shifting mean
                              NORMALISED P(yes) by < 0.05. The raw readout would report a large
                              effect of prompt on the finding decision where there is almost none,
                              which is the same failure as the clamp and reaches a much wider
                              audience.
  NO PROMPT CONFOUND          every phrasing moves both readouts by comparable amounts. Then the
                              confound really is intervention-specific and the paper says so.

GPU. Usage (VM):
  python scripts/readout_prompt_confound.py --n 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FINDING = ("Cardiomegaly", "cardiomegaly")
PROMPTS = {
    "plain": "Does this chest X-ray show cardiomegaly? Answer yes or no.",
    "no_format": "Does this chest X-ray show cardiomegaly?",
    "persona": ("You are an experienced radiologist. Does this chest X-ray show cardiomegaly? "
                "Answer yes or no."),
    "gloss": "Does this chest X-ray show cardiomegaly (an enlarged heart)? Answer yes or no.",
}


def log(*a: object) -> None:
    print(*a, flush=True)


def main() -> None:
    import tracecxr.models  # noqa: F401, PLC0415  # importing registers the real adapters
    from PIL import Image  # noqa: PLC0415
    from tracecxr.core.model import get_model  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medgemma")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="/mnt/fast/clt/readout_prompt_confound.json")
    args = ap.parse_args()

    import pandas as pd  # noqa: PLC0415
    df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
    img_dir = args.data_dir / "images"
    pool = [(str(r["Image Index"]), str(r["Finding Labels"])) for _, r in df.iterrows()
            if not str(r["Image Index"]).startswith("._")
            and (img_dir / str(r["Image Index"])).exists()]
    rng = np.random.default_rng(args.seed)
    pick = [pool[int(i)] for i in rng.permutation(len(pool))[: args.n]]
    log(f"{args.model}: {len(pick)} films (same sample as the evaluation sweep)")

    adapter = get_model(args.model)
    acc = {k: {"raw": [], "norm": [], "mass": []} for k in PROMPTS}
    for k, (name, _) in enumerate(pick):
        img = Image.open(img_dir / name).convert("RGB")
        for pname, prompt in PROMPTS.items():
            r, n, m = adapter.probe_readouts(prompt, img)
            acc[pname]["raw"].append(r)
            acc[pname]["norm"].append(n)
            acc[pname]["mass"].append(m)
        if (k + 1) % 50 == 0:
            log(f"  {k + 1}/{len(pick)}")

    summ = {p: {k: float(np.mean(v)) for k, v in d.items()} for p, d in acc.items()}
    base = summ["plain"]
    log(f"\n=== {args.model}, n={len(pick)} films, {FINDING[1]} ===")
    log(f"  {'phrasing':12s}{'raw':>8}{'norm':>8}{'mass':>8}   {'d raw':>8}{'d norm':>8}")
    confirmed = []
    for p, v in summ.items():
        d_raw, d_norm = v["raw"] - base["raw"], v["norm"] - base["norm"]
        log(f"  {p:12s}{v['raw']:>8.3f}{v['norm']:>8.3f}{v['mass']:>8.3f}   "
            f"{d_raw:>+8.3f}{d_norm:>+8.3f}")
        if p != "plain" and abs(d_raw) > 0.15 and abs(d_norm) < 0.05:
            confirmed.append(p)

    verdict = (f"PROMPT CONFOUND CONFIRMED: {', '.join(confirmed)} shift raw P(yes) by >0.15 while "
               "shifting the decision by <0.05" if confirmed else
               "NO PROMPT CONFOUND: both readouts move together under every phrasing")
    log(f">>> {verdict}")
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "n": len(pick), "prompts": PROMPTS, "summary": summ,
         "confirmed": confirmed, "verdict": verdict}, indent=2))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
