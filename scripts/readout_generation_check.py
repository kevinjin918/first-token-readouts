"""Does the decision readout agree with what the model actually says?

`readout_audit_banked.json` is the paper's central measurement. On 20 cardiomegaly films, clamping
the 80 strongest image features lowers the raw first-token probability of "yes" from 0.921 to
0.628, and P_raw falls below 0.5 on 5 of the 20, while the yes/no-normalised readout P_norm stays
above 0.5 on all 20. The paper reads this as: the clamp moved answer mass, not the decision.

That reading rests on the first token alone. Nobody has looked at what the off-answer mass
(0.372 under `scale -1`) turns into once the model keeps writing. If it is formatting ("**" then
"Yes") or a preamble that ends in yes, P_norm is the right summary. If it is a preamble that ends
in "no", then P_raw was right, the clamp changed the model's answer, and the headline is wrong.

This generates the text, from the same forward passes the readouts are read from.

ARMS, on the 20 banked films (cardiomegaly prompt, in banked order):
  base               the clean film
  black, mean, blur  occlusion of the recorded top-effect block (`occlusion_readout_audit.py`)
  border             black occlusion of the random border block, replayed from that script's RNG
  clamp_scale-1, clamp_ablate+0, clamp_scale-2
                     the film's own top-80 image features (`readout_audit.py`), CLT mode only

RECORDED per film x arm: P_raw, P_norm, answer mass, yes-minus-no logit gap, the top 10 next
tokens, a greedy 64-token continuation, and 10 samples at temperature 1 with no top-k/top-p
truncation (samples from the model's own distribution).

PARSE RULE, fixed before running (`tracecxr.intervention.generation_check.parse_answer`): strip
markdown (* _ # ` >), lowercase, take the first whole-word "yes" or "no"; otherwise NONE. Every
greedy output is also read by hand and any disagreement with the parse is reported.

REPRODUCTION GATE (CLT mode), checked first: base and `scale -1` P_raw must match
`readout_audit_banked.json` to within 0.02 on every film, or the run aborts. In both modes the
first generated token's logits must give the same readouts as the scored forward (within 0.02).

AMENDMENT (2026-10-03, made after the gate failed on film 1 and before any generated text was
read; the aborted run printed nothing past the gate). Base reproduced on all 5 films checked (max
|diff| 0.0033). `scale -1` reproduced to 1e-4 on films 4-5 and 0.001 on film 2, but differed by
0.057 and 0.027 on films 1 and 3, identically under transformers 5.14.1 and 5.18.0. Cause, shown
on film 3: its 80th and 81st image features differ in mean activation by 0.007, and swapping
today's 80th for any of several next-ranked features reproduces the banked 0.5976 exactly. The
banked run selected a different boundary feature; the banked feature sets were not saved, so they
cannot be restored. Single features inside the top 80 move P_raw by up to 0.19 on film 1, so
per-film clamp numbers are sensitive to bf16-level differences at the selection boundary.
Therefore: the base gate stays hard; the clamp arms are re-measured with fresh selection and their
difference from the banked numbers is REPORTED, not gated; the selected (layer, feature) pairs and
the 80th-81st activation gap are saved per film; and the paper takes every clamp number from this
run, not from the banked file. The verdict rules below are unchanged.

SECOND AMENDMENT (2026-10-03, made after the base gate failed on film 7). Unlike the first, this
one was NOT made blind: by then the log had printed, for films 1-6, every arm's P_raw, P_norm and
parsed greedy answer, and they were seen. Film 7 (00000711_003) gives base P_raw 0.6233 against
banked 0.6554 (P_norm 0.998 in both). The plain model with no transcoder loaded gives the same
0.6233, and films 1-6 match it to 1e-4 on the transcoder path, so the difference is the model's own
forward pass under today's software stack (torch 2.14.1, transformers 5.14.1) against the August
one (versions not recorded), largest on films where answer mass is low. The base gate is therefore
reported like the clamp arms. Nothing else changes: same films, arms, seed and verdict rules, and
the run restarts from film 1 so all 20 films come from one process. The gate that the generation
prefill reproduces the scored forward stays hard.

PRE-SPECIFIED, per arm, n = 20. The "P_norm call" is YES iff P_norm >= 0.5.
  DISAGREES        greedy parses NO on >= 3 films where the P_norm call is YES. The clamp changes
                   the model's stated answer and the decision readout misses it: the headline is
                   wrong as stated and must be rewritten around P_raw.
  HEDGE-DOMINATED  otherwise, greedy parses NONE on >= 5 films. Text cannot adjudicate; report the
                   rates and claim neither readout is validated.
  VALIDATED        otherwise, greedy matches the P_norm call on >= 18 of 20 films.
  MIXED            anything else.
Reported for every arm whatever the verdict: greedy agreement with the P_raw call (P_raw >= 0.5),
and the sampled YES/NO/NONE rates beside mean P_raw and P_norm.

Laptop (no CLT; base and occlusion arms only):
  python scripts/readout_generation_check.py --no-clt --device mps --out results/clt_scale/...
GPU (all arms):
  python scripts/readout_generation_check.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tracecxr.intervention.generation_check import (
    arm_verdict,
    call,
    parse_answer,
    replay_border_blocks,
    tally,
)

GRID, BLOCK = 16, 2
NB = GRID // BLOCK
PROMPT = "Does this chest X-ray show cardiomegaly? Answer yes or no."
N_FEATS = 80
OCCLUSION_ARMS = ("base", "black", "mean", "blur", "border")
CLAMP_OPS = {"clamp_scale-1": ("scale", -1.0), "clamp_ablate+0": ("ablate", 0.0),
             "clamp_scale-2": ("scale", -2.0)}
BANKED_KEY = {"clamp_scale-1": "cardiomegaly_scale-1", "clamp_ablate+0": "cardiomegaly_ablate+0",
              "clamp_scale-2": "cardiomegaly_scale-2"}
TOL = 0.02


def log(*a: object) -> None:
    print(*a, flush=True)


def block_slice(a, b):
    h, w = a.shape[:2]
    br, bc = divmod(int(b), NB)
    return (slice(int(br * BLOCK * h / GRID), int((br + 1) * BLOCK * h / GRID)),
            slice(int(bc * BLOCK * w / GRID), int((bc + 1) * BLOCK * w / GRID)))


def fill(arr, b, how):
    """Identical to `occlusion_readout_audit.fill`."""
    a = arr.copy()
    ys, xs = block_slice(a, b)
    if how == "black":
        a[ys, xs] = 0
    elif how == "mean":
        a[ys, xs] = int(arr.mean())
    elif how == "blur":
        a[ys, xs] = np.array(Image.fromarray(arr[ys, xs]).filter(ImageFilter.GaussianBlur(12)))
    return Image.fromarray(a)


def main() -> None:  # noqa: PLR0915
    import torch  # noqa: PLC0415
    from tracecxr.core.config import MODELS  # noqa: PLC0415
    from tracecxr.models._base import resolve_yes_no_token_ids, yes_probability  # noqa: PLC0415
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/clt_scale"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--no-clt", action="store_true", help="base and occlusion arms only")
    ap.add_argument("--ckpt", default="/mnt/fast/clt/orig2048/clt_ckpt.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-films", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--arms", default=None, help="comma-separated subset, for debugging")
    ap.add_argument("--out", type=Path, default=Path("/mnt/fast/clt/readout_generation_check.json"))
    args = ap.parse_args()

    banked = json.loads((args.results / "readout_audit_banked.json").read_text())["fire_rows"]
    occl = json.loads((args.results / "occlusion_readout_audit.json").read_text())["rows"]
    attn = json.loads((args.results / "attn_causal_map.json").read_text())["A"]["records"]
    films = [r["image"] for r in banked]
    assert films == [r["image"] for r in occl] == [r["image"] for r in attn[: len(films)]], \
        "banked files disagree on film order"
    top_block = {r["image"]: int(r["top_block"]) for r in occl}
    border = dict(zip(films, replay_border_blocks(len(films)), strict=True))
    banked_by = {r["image"]: r for r in banked}
    films = films[: args.n_films]

    arms = list(OCCLUSION_ARMS) + ([] if args.no_clt else list(CLAMP_OPS))
    if args.arms:
        arms = [a for a in args.arms.split(",") if a in arms]

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mid = MODELS["medgemma"].locator
    cmg = None
    if args.no_clt:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForImageTextToText.from_pretrained(
            mid, dtype=torch.bfloat16).to(dev).eval()
    else:
        from tracecxr.intervention import ClampedMedGemma, FeatureEdit  # noqa: PLC0415
        from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

        res = json.loads((args.results / "clt_live_result.json").read_text())
        layers = list(res["layers"])
        cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                        k=res.get("k", 32), n_layers=len(layers),
                        adam_8bit=bool(res.get("adam_8bit", False)), device=dev)
        cmg = ClampedMedGemma(args.ckpt, layers, clt_config=cfg, model_id=mid, device=dev)
        cmg._load()
        proc, model, clt = cmg._state["proc"], cmg._state["model"], cmg._state["clt"]
    tok = getattr(proc, "tokenizer", proc)
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    img_tok = int(model.config.image_token_index)
    log(f"loaded MedGemma on {dev} ({'no CLT' if args.no_clt else 'with CLT'}); arms {arms}")

    def readouts(row):
        row = row.float()
        pv = torch.softmax(row, -1)
        y, n = pv[yes_ids].max(), pv[no_ids].max()
        top = torch.topk(pv, 10)
        return {"raw": float(y), "norm": float(yes_probability(row, yes_ids, no_ids)),
                "mass": float(y + n),
                "gap": float(row[yes_ids].max() - row[no_ids].max()),
                "top10": [[int(i), tok.decode([int(i)]), float(p)]
                          for p, i in zip(top.values, top.indices, strict=True)]}

    def plain_generate(image):
        inp = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": PROMPT}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(dev)
        plen = inp["input_ids"].shape[1]
        with torch.inference_mode():
            scored = model(**inp).logits[0, plen - 1].float()
            g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                               top_k=None, top_p=None, output_logits=True,
                               return_dict_in_generate=True)
            s = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                               temperature=1.0, top_k=0, top_p=1.0,
                               num_return_sequences=args.n_samples)
        return scored, {"greedy": tok.decode(g.sequences[0, plen:], skip_special_tokens=True),
                        "samples": tok.batch_decode(s[:, plen:], skip_special_tokens=True),
                        "first_logits": g.logits[0][0].float()}

    def mean_feats(image):
        """Identical to `readout_audit.mean_feats`: CLT activations averaged over image tokens."""
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

        hs = [cmg._state["layers_mod"][L].mlp.register_forward_hook(mk(L)) for L in layers]
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

    def topn(score, n=N_FEATS):
        flat = np.argsort(-score, axis=None)[:n]
        return [(layers[int(i // score.shape[1])], int(i % score.shape[1])) for i in flat]

    def run_arm(image, edits):
        if cmg is None:
            scored, gen = plain_generate(image)
        else:
            scored = cmg.decision_logits_row(prompt=PROMPT, image=image, edits=edits)
            gen = cmg.generate(prompt=PROMPT, image=image, edits=edits,
                               max_new_tokens=args.max_new_tokens, num_samples=args.n_samples)
        ro, first = readouts(scored), readouts(gen["first_logits"])
        diff = max(abs(ro["raw"] - first["raw"]), abs(ro["norm"] - first["norm"]))
        assert diff < TOL, f"generation prefill does not match the scored forward ({diff:.4f})"
        return {**ro, "first_token_diff": diff, "greedy": gen["greedy"],
                "greedy_parsed": parse_answer(gen["greedy"]), "samples": gen["samples"],
                "samples_parsed": [parse_answer(t) for t in gen["samples"]]}

    torch.manual_seed(args.seed)
    img_dir = args.data_dir / "images"
    rows: list[dict] = []
    t0 = time.time()
    for k, name in enumerate(films):
        pil = Image.open(img_dir / name).convert("RGB")
        arr = np.array(pil)
        rec: dict = {"image": name, "top_block": top_block[name], "border_block": border[name]}
        pairs = None
        for arm in arms:
            edits: list = []
            image = pil
            if arm in ("black", "mean", "blur"):
                image = fill(arr, top_block[name], arm)
            elif arm == "border":
                image = fill(arr, border[name], "black")
            elif arm in CLAMP_OPS:
                if pairs is None:
                    score = mean_feats(pil)
                    pairs = topn(score)
                    srt = np.sort(score, axis=None)[::-1]
                    rec["clamp_pairs"] = [[int(L), int(f)] for L, f in pairs]
                    rec["boundary_gap"] = float(srt[N_FEATS - 1] - srt[N_FEATS])
                op, amt = CLAMP_OPS[arm]
                edits = [FeatureEdit(layer=L, feature=f, op=op, amount=amt, positions="image")
                         for L, f in pairs]
            rec[arm] = run_arm(image, edits)
            # reproduction gate against the banked numbers, before anything else is trusted
            ref = (banked_by[name]["base_raw"] if arm == "base" else
                   banked_by[name][BANKED_KEY[arm]]["raw"] if arm in CLAMP_OPS else
                   next(r for r in occl if r["image"] == name)[arm]["raw"])
            rec[arm]["banked_raw"] = ref  # reported, not gated: see both AMENDMENTs
        rows.append(rec)
        log(f"[{k + 1}/{len(films)} {time.time() - t0:.0f}s] {name}  " + "  ".join(
            f"{a}: {rec[a]['raw']:.2f}/{rec[a]['norm']:.2f} {rec[a]['greedy_parsed']}"
            for a in arms))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"partial": True, "rows": rows}, indent=1))

    summary = {}
    log(f"\n=== n={len(rows)} films; greedy answer vs each readout's call ===")
    log(f"  {'arm':16s}{'P_raw':>7s}{'P_norm':>8s}{'mass':>7s}  greedy Y/N/0   "
        f"=raw  =norm   sampled Y/N/0      verdict")
    for arm in arms:
        raw = [r[arm]["raw"] for r in rows]
        norm = [r[arm]["norm"] for r in rows]
        greedy = [r[arm]["greedy_parsed"] for r in rows]
        g = tally(greedy)
        s = tally(a for r in rows for a in r[arm]["samples_parsed"])
        ns = max(sum(s.values()), 1)
        agree_raw = sum(call(p) == a for p, a in zip(raw, greedy, strict=True))
        agree_norm = sum(call(p) == a for p, a in zip(norm, greedy, strict=True))
        verdict = arm_verdict(norm, greedy)
        summary[arm] = {
            "raw": float(np.mean(raw)), "norm": float(np.mean(norm)),
            "mass": float(np.mean([r[arm]["mass"] for r in rows])),
            "gap_median": float(np.median([r[arm]["gap"] for r in rows])),
            "banked_raw_maxdiff": float(max(abs(r[arm]["raw"] - r[arm]["banked_raw"])
                                            for r in rows)),
            "flips_raw": sum(call(p) == "NO" for p in raw),
            "flips_norm": sum(call(p) == "NO" for p in norm),
            "greedy": g, "agree_raw": agree_raw, "agree_norm": agree_norm,
            "sampled": s, "sampled_yes_rate": s["YES"] / ns,
            "sampled_yes_given_answer": s["YES"] / max(s["YES"] + s["NO"], 1),
            "verdict": verdict}
        log(f"  {arm:16s}{np.mean(raw):>7.3f}{np.mean(norm):>8.3f}"
            f"{summary[arm]['mass']:>7.3f}  {g['YES']:>2d}/{g['NO']:>2d}/{g['NONE']:>2d}"
            f"      {agree_raw:>2d}   {agree_norm:>2d}    "
            f"{s['YES'] / ns:.2f}/{s['NO'] / ns:.2f}/{s['NONE'] / ns:.2f}   {verdict}")

    args.out.write_text(json.dumps(
        {"n": len(rows), "arms": arms, "mode": "no-clt" if args.no_clt else "clt",
         "versions": {"torch": torch.__version__, "transformers": transformers_version},
         "device": dev, "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens,
         "seed": args.seed, "summary": summary, "rows": rows}, indent=1))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
