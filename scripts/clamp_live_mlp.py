"""E8: does the steering result depend on freezing the MLPs in the prompt?

The clamp of E2-E5 (``ClampedMedGemma``, mode ``"replace"``) encodes the transcoder features once,
on the unedited prefill, and substitutes ``decode(edited) + error`` for every captured MLP output in
the prompt. Nothing is edited at text positions, so there, including the readout position, the MLP
outputs keep their unedited values: the edit reaches the first token only through attention.
Tokens after the first are computed by the live model. The adversarial audit of 2026-10-04 noted
that the readouts and the written answers are therefore computed in different regimes, so E4's
headline (V1: P_norm MISREADS in all three clamp arms) could be a property of that asymmetry.

This run adds the same edit to the live model instead (mode ``"add"``): every MLP runs from its
live input on the prefill, and the change ``decode(edited) - decode(clean)``, nonzero only at image
positions, is added to its output. With no edit this is the plain model. The first token and every
later token are then computed by the same live model.

FILMS. E4's 40 false-positive films (fp_banked + fp_fresh), each with E4's saved clamp feature set
(``clamp_pairs``, its top-80 image features), read from
``results/clt_scale/readout_sampled_answers.json``. The script refuses any other set: the sha256
of the sorted ``group/image`` names must be
f42396a8aaed4acd73d32c6c4fcba67f8d38934b43e4de06a9e57160ddba6ae3.

ARMS, all mode "add": base (no edit), clamp_scale-1, clamp_ablate+0, clamp_scale-2. Same prompt,
greedy answer and 50 samples at T = 1 (top_k = 0, top_p = 1), 128 new tokens, E2's parse rule, and
E4's per-film seed (20261004 + 1000 * group index + E4's film index). E4's first-token gate applies
(the generated first-token distribution must match the scored forward within 0.02).

PRE-SPECIFIED, fixed before running (commit = pre-registration).
The rule is E4's, run by E4's own code (``summarize_arm`` in ``readout_sampled_answers.py``,
unchanged): per film diff = (R_base - R_arm) - (s_base - s_arm), film-bootstrap interval of the
mean (10,000 resamples), MISREADS if it excludes 0 and |mean| > 0.05, TRACKS if inside [-0.05,
0.05], else UNRESOLVED. Base is this run's base arm.
E8 verdict, on P_norm against s over the three clamp arms:
  INCONCLUSIVE           if in every arm the interval of the sampled drop (s_base - s_arm)
                         contains 0: the live edit does not change the answers.
  HOLDS WITH LIVE MLPS   otherwise, if P_norm MISREADS with a negative mean (the readout falls
                         less than the answers, E4's direction) in all three arms.
  FROZEN-MLP ARTIFACT    otherwise, if P_norm TRACKS in all three arms.
  PARTIAL                otherwise.
What the paper will say, fixed now: HOLDS -> the result does not depend on freezing the MLPs.
FROZEN-MLP ARTIFACT -> the steering result is reported as a property of the frozen-MLP clamp and
the title, abstract and contributions are narrowed to that clamp. INCONCLUSIVE -> the paper
states that the suppression's effect on the written answers itself depends on freezing the MLPs,
in the main text, and narrows the steering claims to the frozen-MLP clamp as for the artifact
verdict. PARTIAL -> reported arm by arm, and the frozen-MLP caveat stays in the main text.
Whatever the verdict, the result is reported in the paper.
Reported, not verdicts: the same rule for P_raw and for the answered-only rate c; per arm mean
P_raw, P_norm, answer mass and s; the opening-word shares and P(yes | opening word); per film, this
run's P_norm drop and s drop minus E4's (live minus frozen); the base arm's largest difference
from E4's base readouts.

GPU. Usage (VM):
  python scripts/clamp_live_mlp.py
  python scripts/clamp_live_mlp.py --analyze-only results/clt_scale/clamp_live_mlp.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from tracecxr.intervention.answer_runner import PROMPTS
from tracecxr.intervention.generation_check import bootstrap_ci, yes_rate

FILM_SHA = "f42396a8aaed4acd73d32c6c4fcba67f8d38934b43e4de06a9e57160ddba6ae3"
ARMS = ("base", "clamp_scale-1", "clamp_ablate+0", "clamp_scale-2")
MODE = "add"


def log(*a: object) -> None:
    print(*a, flush=True)


def e4_module() -> object:
    path = Path(__file__).with_name("readout_sampled_answers.py")
    spec = importlib.util.spec_from_file_location("readout_sampled_answers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e4_films(e4: dict) -> list[tuple[int, dict]]:
    """E4's false-positive rows with E4's film index k (its enumeration over all groups)."""
    groups = ("fp_banked", "fp_fresh", "tp_cardiomegaly", "tp_effusion", "tp_pneumothorax")
    order = [(g, f["image"]) for g in groups for f in e4["draw"].get(g, [])]
    k_of = {key: k for k, key in enumerate(order)}
    rows = [r for r in e4["rows"] if r["group"].startswith("fp_")]
    names = sorted(f"{r['group']}/{r['image']}" for r in rows)
    sha = hashlib.sha256("\n".join(names).encode()).hexdigest()
    if sha != FILM_SHA:
        raise SystemExit(f"film set {sha} is not the registered one")
    return [(k_of[(r["group"], r["image"])], r) for r in rows]


# ---------------------------------------------------------------------------------------------
# analysis (pure; runs on the saved JSON)
# ---------------------------------------------------------------------------------------------
def analyze(rows: list[dict], e4_rows: list[dict], *, n_boot: int = 10000) -> dict:
    mod = e4_module()
    summary = {arm: mod.summarize_arm(rows, arm, n_boot=n_boot) for arm in ARMS}
    clamps = ARMS[1:]
    no_effect = all(summary[a]["s_drop"][1] <= 0 <= summary[a]["s_drop"][2] for a in clamps)
    vs = [summary[a]["norm_vs_s"] for a in clamps]
    if no_effect:
        verdict = "INCONCLUSIVE"
    elif all(v["verdict"] == "MISREADS" and v["diff"][0] < 0 for v in vs):
        verdict = "HOLDS WITH LIVE MLPS"
    elif all(v["verdict"] == "TRACKS" for v in vs):
        verdict = "FROZEN-MLP ARTIFACT"
    else:
        verdict = "PARTIAL"

    frozen = {(r["group"], r["image"]): r for r in e4_rows}
    compare: dict = {}
    for a in clamps:
        dn, ds = [], []
        for r in rows:
            f = frozen[(r["group"], r["image"])]
            live_b, live_a = r["arms"]["base"], r["arms"][a]
            fr_b, fr_a = f["arms"]["base"], f["arms"][a]
            dn.append((live_b["norm"] - live_a["norm"]) - (fr_b["norm"] - fr_a["norm"]))
            ds.append((yes_rate(live_b["samples_parsed"]) - yes_rate(live_a["samples_parsed"]))
                      - (yes_rate(fr_b["samples_parsed"]) - yes_rate(fr_a["samples_parsed"])))
        compare[a] = {"norm_drop_live_minus_frozen": bootstrap_ci(np.array(dn), n_boot=n_boot),
                      "s_drop_live_minus_frozen": bootstrap_ci(np.array(ds), n_boot=n_boot)}
    base_diff = max(max(abs(r["arms"]["base"][k] - frozen[(r["group"], r["image"])]["arms"]
                            ["base"][k]) for k in ("raw", "norm")) for r in rows)
    return {"summary": summary, "verdict": verdict,
            "arm_verdicts": {a: summary[a]["norm_vs_s"]["verdict"] for a in clamps},
            "raw_verdicts": {a: summary[a]["raw_vs_s"]["verdict"] for a in clamps},
            "live_vs_frozen": compare, "base_max_diff_from_e4": base_diff}


def report(res: dict) -> None:
    log(f"  {'arm':15s}{'P_raw':>7s}{'P_norm':>7s}{'mass':>7s}{'s':>7s}  {'s drop [95% CI]':>22s}"
        f"  {'raw-s':>18s}  {'norm-s':>18s}")
    for arm, a in res["summary"].items():
        sd = a.get("s_drop")
        line = (f"  {arm:15s}{a['raw']:7.3f}{a['norm']:7.3f}{a['mass']:7.3f}{a['s_mean'][0]:7.3f}  "
                + (f"{sd[0]:+.3f} [{sd[1]:+.3f},{sd[2]:+.3f}]" if sd else " " * 22))
        for r in ("raw", "norm"):
            if f"{r}_vs_s" in a:
                line += f"  {a[f'{r}_vs_s']['diff'][0]:+.3f} {a[f'{r}_vs_s']['verdict']:>11s}"
        log(line)
    for arm, c in res["live_vs_frozen"].items():
        n, s = c["norm_drop_live_minus_frozen"], c["s_drop_live_minus_frozen"]
        log(f"  live - frozen, {arm}: P_norm drop {n[0]:+.3f} [{n[1]:+.3f},{n[2]:+.3f}]"
            f"  s drop {s[0]:+.3f} [{s[1]:+.3f},{s[2]:+.3f}]")
    log(f"  base arm, largest |readout - E4 base|: {res['base_max_diff_from_e4']:.4f}")
    log(f"\nE8 VERDICT: {res['verdict']}  (P_norm per arm: {res['arm_verdicts']})")


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
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--analyze-only", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("/mnt/fast/clt/clamp_live_mlp.json"))
    args = ap.parse_args()

    e4 = json.loads((args.results / "readout_sampled_answers.json").read_text())
    films = e4_films(e4)
    e4_rows = [r for _, r in films]
    if args.analyze_only:
        res = analyze(json.loads(args.analyze_only.read_text())["rows"], e4_rows)
        report(res)
        return

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from tracecxr.intervention.answer_runner import AnswerRunner  # noqa: PLC0415
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    mod = e4_module()
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
        for arm in ARMS:
            edits = [] if arm == "base" else runner.edits(f["clamp_pairs"], *mod.CLAMP_OPS[arm])
            rec["arms"][arm] = runner.answer(prompt, pil, edits, n_samples=args.n_samples,
                                             max_new_tokens=args.max_new_tokens, mode=MODE)
        rows.append(rec)
        log(f"[{i + 1}/{len(films)} {time.time() - t0:.0f}s] {g} {name}  " + "  ".join(
            f"{a}: {v['raw']:.2f}/{v['norm']:.2f} s={yes_rate(v['samples_parsed']):.2f}"
            for a, v in rec["arms"].items()))
        prior["rows"] = rows
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prior, indent=1))

    res = analyze(rows, e4_rows)
    report(res)
    prior.update({"versions": {"torch": torch.__version__, "transformers": transformers_version},
                  "mode": MODE, "n_samples": args.n_samples,
                  "max_new_tokens": args.max_new_tokens, "seed": seed, **res})
    args.out.write_text(json.dumps(prior, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
