"""E9: with live MLPs, at a strength that leaves the model answering, does P_norm misread?

E8 (``clamp_live_mlp.py``) added E4's three edits to the live model. On the films inspected while
it ran, the live edits mostly stopped the model answering: most samples gave neither yes nor no,
said the model could not see an image, or were not language. E8 therefore cannot compare the
readouts with the answers at the answer change E4 measured, which is the comparison its question
needs. This run weakens the live edit until the model keeps answering.

FILMS AND FEATURES. As E8: E4's 40 false-positive films, each with E4's saved feature set (its
top-80 image features), refused unless the film set's sha256 is E8's.

ARMS, all mode "add" (the edit's change added to the live MLPs, as E8): base (no edit) and partial
suppression, every feature in the set scaled by c (op "scale", amount c) at image positions, for
c in DOSES = (0.9, 0.75, 0.6, 0.45, 0.3, 0.15). E8's ablation is c = 0. Same prompt, greedy answer,
50 samples at T = 1 (top_k = 0, top_p = 1), 128 new tokens, E2's parse rule, E4's per-film seed,
E4's first-token gate.

PRE-SPECIFIED, fixed before running (commit = pre-registration).
Non-answer share of an arm: the pooled share of samples that give neither yes nor no under E2's
rule or say they cannot see an image (``_NO_IMAGE`` in ``answer_text_checks.py``). TAU, the most
non-answering E4 allowed: the largest non-answer share among E4's arms on these films, computed
from E4's file (0.167, under clamp_scale-2).
An arm is ELIGIBLE if its non-answer share is at most TAU and the film-bootstrap interval of its
drop in s (s_base - s_arm, 10,000 resamples) lies above 0.05, E4's margin.
The MATCHED DOSE is the eligible arm whose mean s drop is closest to E4's under clamp_scale-1 (the
headline arm, frozen MLPs); a tie goes to the larger c (the weaker edit).
The rule is E4's, run by E4's own code (``summarize_arm``, unchanged): per film diff = (R_base -
R_arm) - (s_base - s_arm), MISREADS if the interval excludes 0 and |mean| > 0.05, TRACKS if it lies
inside [-0.05, 0.05], else UNRESOLVED.
E9 verdict, on P_norm against s at the matched dose:
  NO MATCHED DOSE   no arm is eligible.
  HOLDS LIVE        P_norm MISREADS with a negative mean (it falls less than the answers, E4).
  REVERSES LIVE     P_norm MISREADS with a positive mean (it falls more than the answers).
  TRACKS LIVE       P_norm TRACKS.
  UNRESOLVED        otherwise.
What the paper will say, fixed now:
  HOLDS LIVE -> the main text states that the readout understates the answer change with the
    live model at a strength matched to E4's answer change, and the frozen-MLP threat is no longer
    listed as open.
  TRACKS LIVE or REVERSES LIVE -> the understatement is a property of the frozen-MLP clamp: the
    title, abstract and contributions are narrowed to that clamp, and the main text states what
    the live model does at the matched strength.
  UNRESOLVED -> reported, and the frozen-MLP caveat stays in the main text as the main open threat.
  NO MATCHED DOSE -> the main text states that the live edit changes the answers only by stopping
    the model answering, and the steering claims are narrowed to the frozen-MLP clamp.
Whatever the verdict, the result is reported in the paper.
Reported, not verdicts: per dose, the non-answer share, mean P_raw, P_norm, answer mass and s, the
s drop, and E4's rule for P_norm and P_raw against s and against the answered-only rate c; the
base arm's largest difference from E8's base readouts.

GPU. Usage (VM):
  python scripts/clamp_live_dose.py
  python scripts/clamp_live_dose.py --analyze-only results/clt_scale/clamp_live_dose.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from tracecxr.intervention.answer_runner import PROMPTS
from tracecxr.intervention.generation_check import yes_rate

DOSES = (0.9, 0.75, 0.6, 0.45, 0.3, 0.15)
ARMS = ("base", *(f"live_scale{c}" for c in DOSES))
MODE = "add"
MARGIN = 0.05


def log(*a: object) -> None:
    print(*a, flush=True)


def sibling(name: str) -> object:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nonanswer(rec: dict, text: object) -> list[bool]:
    return [p == "NONE" or bool(text._NO_IMAGE.search(text._clean(t)))  # type: ignore[attr-defined]
            for t, p in zip(rec["samples"], rec["samples_parsed"], strict=True)]


def tau(e4_rows: list[dict], text: object) -> float:
    """The largest pooled non-answer share among E4's arms on these films."""
    arms = e4_rows[0]["arms"]
    return max(float(np.mean([x for r in e4_rows for x in nonanswer(r["arms"][a], text)]))
               for a in arms)


# ---------------------------------------------------------------------------------------------
# analysis (pure; runs on the saved JSON)
# ---------------------------------------------------------------------------------------------
def analyze(rows: list[dict], e4_rows: list[dict], e8_rows: list[dict] | None,
            *, n_boot: int = 10000) -> dict:
    e4mod, text = sibling("readout_sampled_answers"), sibling("answer_text_checks")
    t = tau(e4_rows, text)
    e4_target = e4mod.summarize_arm(e4_rows, "clamp_scale-1", n_boot=n_boot)["s_drop"][0]
    summary = {arm: e4mod.summarize_arm(rows, arm, n_boot=n_boot) for arm in ARMS}
    doses: dict = {}
    for c, arm in zip(DOSES, ARMS[1:], strict=True):
        a = summary[arm]
        na = float(np.mean([x for r in rows for x in nonanswer(r["arms"][arm], text)]))
        doses[arm] = {"c": c, "nonanswer": na, "s_drop": a["s_drop"],
                      "eligible": na <= t and a["s_drop"][1] > MARGIN,
                      "norm_vs_s": a["norm_vs_s"], "raw_vs_s": a["raw_vs_s"],
                      "norm_vs_c": a["norm_vs_c"], "raw_vs_c": a["raw_vs_c"]}
    eligible = [arm for arm in ARMS[1:] if doses[arm]["eligible"]]
    matched = None
    if eligible:  # ARMS run from the weakest edit, so min() breaks ties toward the larger c
        matched = min(eligible, key=lambda arm: abs(doses[arm]["s_drop"][0] - e4_target))
    if matched is None:
        verdict = "NO MATCHED DOSE"
    else:
        v = doses[matched]["norm_vs_s"]
        if v["verdict"] == "MISREADS":
            verdict = "HOLDS LIVE" if v["diff"][0] < 0 else "REVERSES LIVE"
        elif v["verdict"] == "TRACKS":
            verdict = "TRACKS LIVE"
        else:
            verdict = "UNRESOLVED"
    base_diff = None
    if e8_rows is not None:
        e8 = {(r["group"], r["image"]): r for r in e8_rows}
        base_diff = max(abs(r["arms"]["base"][k] - e8[(r["group"], r["image"])]["arms"]["base"][k])
                        for r in rows for k in ("raw", "norm"))
    return {"summary": summary, "doses": doses, "tau": t, "e4_target_s_drop": e4_target,
            "eligible": eligible, "matched": matched, "verdict": verdict,
            "base_max_diff_from_e8": base_diff}


def report(res: dict) -> None:
    log(f"  tau {res['tau']:.3f}; E4 clamp_scale-1 s drop {res['e4_target_s_drop']:.3f}")
    log(f"  {'arm':18s}{'P_raw':>7s}{'P_norm':>7s}{'mass':>7s}{'s':>7s}{'non-ans':>8s}"
        f"  {'s drop [95% CI]':>22s}  {'norm-s':>18s}  {'norm-c':>18s}  elig")
    for arm, a in res["summary"].items():
        d = res["doses"].get(arm)
        line = f"  {arm:18s}{a['raw']:7.3f}{a['norm']:7.3f}{a['mass']:7.3f}{a['s_mean'][0]:7.3f}"
        if d:
            sd = d["s_drop"]
            line += (f"{d['nonanswer']:8.3f}  {sd[0]:+.3f} [{sd[1]:+.3f},{sd[2]:+.3f}]"
                     f"  {d['norm_vs_s']['diff'][0]:+.3f} {d['norm_vs_s']['verdict']:>11s}"
                     f"  {d['norm_vs_c']['diff'][0]:+.3f} {d['norm_vs_c']['verdict']:>11s}"
                     f"  {'yes' if d['eligible'] else 'no'}")
        log(line)
    if res["base_max_diff_from_e8"] is not None:
        log(f"  base arm, largest |readout - E8 base|: {res['base_max_diff_from_e8']:.4f}")
    log(f"\nE9 VERDICT: {res['verdict']}  (matched dose: {res['matched']})")


# ---------------------------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/clt_scale"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--ckpt", default="/mnt/fast/clt/orig2048/clt_ckpt.pt")
    ap.add_argument("--e8", type=Path, default=Path("/mnt/fast/clt/clamp_live_mlp.json"))
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--analyze-only", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("/mnt/fast/clt/clamp_live_dose.json"))
    args = ap.parse_args()

    e8mod = sibling("clamp_live_mlp")
    e4 = json.loads((args.results / "readout_sampled_answers.json").read_text())
    films = e8mod.e4_films(e4)
    e4_rows = [r for _, r in films]
    e8_rows = json.loads(args.e8.read_text())["rows"] if args.e8.exists() else None
    if args.analyze_only:
        report(analyze(json.loads(args.analyze_only.read_text())["rows"], e4_rows, e8_rows))
        return

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from tracecxr.intervention.answer_runner import AnswerRunner  # noqa: PLC0415
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    mod = sibling("readout_sampled_answers")
    seed = int(e4["seed"])
    runner = AnswerRunner.load(args.results, args.ckpt)
    log(f"loaded MedGemma + CLT ({len(runner.layers)} layers); {len(films)} films; mode {MODE}")
    img_dir = args.data_dir / "images"
    prior = json.loads(args.out.read_text()) if args.out.exists() else {"rows": []}
    rows: list[dict] = list(prior["rows"])
    done = {(r["group"], r["image"]) for r in rows}
    t0 = time.time()
    for i, (k, f) in enumerate(films):
        g, name = f["group"], f["image"]
        if (g, name) in done:
            continue
        torch.manual_seed(seed + 1000 * mod.GROUPS.index(g) + k)
        prompt = PROMPTS[f["finding"]]
        pil = Image.open(img_dir / name).convert("RGB")
        rec = {"group": g, "image": name, "finding": f["finding"],
               "clamp_pairs": f["clamp_pairs"], "arms": {}}
        for arm, c in zip(ARMS, (None, *DOSES), strict=True):
            edits = [] if c is None else runner.edits(f["clamp_pairs"], "scale", c)
            rec["arms"][arm] = runner.answer(prompt, pil, edits, n_samples=args.n_samples,
                                             max_new_tokens=args.max_new_tokens, mode=MODE)
        rows.append(rec)
        log(f"[{i + 1}/{len(films)} {time.time() - t0:.0f}s] {g} {name}  " + "  ".join(
            f"{a[10:] or a}: {v['norm']:.2f} s={yes_rate(v['samples_parsed']):.2f}"
            for a, v in rec["arms"].items()))
        prior["rows"] = rows
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prior, indent=1))

    res = analyze(rows, e4_rows, e8_rows)
    report(res)
    prior.update({"versions": {"torch": torch.__version__, "transformers": transformers_version},
                  "mode": MODE, "doses": list(DOSES), "n_samples": args.n_samples,
                  "max_new_tokens": args.max_new_tokens, "seed": seed, **res})
    args.out.write_text(json.dumps(prior, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
