"""Does scoring a closed yes/no evaluation by first-token probability change what it concludes?

`readout_changes_evaluation.py` compared two first-token readouts (P_raw, P_norm) on 200 random
films and found AUCs within 0.023. That sample had 11, 20 and 8 positives, too few for intervals,
and neither readout was checked against what the model says. Under feature suppression the two
first-token readouts and the generated answer come apart (`readout_generation_check.md`), so the
reference here is the sampled answer, as in `readout_sampled_answers.py`.

FILMS, stratified, drawn before any model runs (same draw for every model): for each finding
(cardiomegaly, effusion, pneumothorax), up to 100 patients carrying the NIH label and the same
number not carrying it, one film per patient within each set, seeded order.

SCORES per film x finding, all from one model and one prompt (the paper's three yes/no questions):
  raw     P_raw, max yes-form probability over the full vocabulary
  norm    P_norm, yes vs no
  s       fraction of --n-samples answers (T = 1, top_k = 0, top_p = 1, no repetition penalty)
          that parse YES with E2's rule
  greedy  the greedy answer: YES 1, NO 0, NONE 0.5
Hard gate: the first generated token's readouts match the scored forward within 0.02.

PRE-SPECIFIED, fixed before running (commit = pre-registration). AUC against the label, film
bootstrap (2,000 resamples, paired across scores).
E6 per model, over the six differences AUC(raw) - AUC(s) and AUC(norm) - AUC(s), three findings:
    READOUT CHANGES THE EVALUATION   some |difference| > 0.03 with an interval excluding 0
    INNOCUOUS                        every interval inside [-0.03, 0.03]
    UNRESOLVED                       otherwise
E6-rank across two models (`--compare a.json b.json`, same films): per finding, the interval of
AUC_A - AUC_B under each score, resampling films jointly.
    RANKING REVERSES                 for some finding, raw or norm excludes 0 on one side and s
                                     excludes 0 on the other
    RANKING DIFFERS IN SIGNIFICANCE  otherwise, for some finding exactly one of (raw or norm, s)
                                     excludes 0
    SAME RANKING                     otherwise
Reported: per finding, calls that disagree (P >= 0.5 against the sampled majority, and against
the greedy answer), accuracy of each call against the label, and opening words by label.

AMENDMENT 1 (2026-10-03, committed before any second-model run; no second-model output exists).
The second model is CheXagent-2-3b (StanfordAIMI/CheXagent-2-3b, revision 8f19b53, MIT), not
Lingshu-7B: the author chose a chest-radiograph model from an independent group over a general
medical one. Lingshu stays runnable. CheXagent ships its own modeling code, pinned by its model card
to transformers 4.40.0, so it runs in a separate environment (``~/venv-chexagent``) and is prompted
in its model card's format: system turn "You are a helpful assistant.", then the image, passed by
file path through ``tokenizer.from_list_format``, then the same question as for MedGemma. Its own
code loads and preprocesses the image. Its template writes every assistant turn as
``<|assistant|>`` + newline + text but its generation prompt stops at ``<|assistant|>``, so the
newline is appended and the first-token readouts are taken at the first content token, as for
MedGemma, whose prompt ends ``<start_of_turn>model`` + newline. Each row records the model's own
top token at the position before (``template_nl_top``); it should be the newline. Films, scores,
parse rule, sampling settings, the first-token gate and every decision rule are unchanged.
MedGemma's run is not affected.

AMENDMENT 2 (2026-10-03, after 25 of 500 CheXagent pairs; the run was stopped by the check that
amendment 1 specified). The recorded ``template_nl_top`` was never the newline: at ``<|assistant|>``
the model's own top token is the answer itself ("Yes" on 22 of the 25 pairs, "No" on 3). The
appended newline was therefore off the model's format and is dropped: the prompt is the model
card's, ending at ``<|assistant|>``, and the readouts are taken there. The 25 pairs are kept in
``readout_eval_answers_chexagent_aborted_nl.json`` and not used. Nothing else changes.

GPU. Usage (VM):
  python scripts/readout_eval_answers.py --model medgemma
  ~/venv-chexagent/bin/python scripts/readout_eval_answers.py --model chexagent
  python scripts/readout_eval_answers.py --model lingshu      # runnable, not part of the study
  python scripts/readout_eval_answers.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tracecxr.intervention.answer_runner import LABELS, PROMPTS, patient
from tracecxr.intervention.generation_check import (
    auc,
    auc_bootstrap,
    call,
    eval_verdict,
    opening_word,
    parse_answer,
    yes_rate,
)

MODELS = {  # name -> (HF id, pinned revision or None for the registry's MedGemma)
    "medgemma": (None, None),
    "lingshu": ("lingshu-medical-mllm/Lingshu-7B", "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"),
    "chexagent": ("StanfordAIMI/CheXagent-2-3b", "8f19b53a2eceda4c33b0acec6c81fbc293ad80d0"),
}
CHEXAGENT_SYSTEM = "You are a helpful assistant."  # the model card's
SCORES = ("raw", "norm", "s", "greedy")
GREEDY_SCORE = {"YES": 1.0, "NO": 0.0, "NONE": 0.5}
TOL = 0.02


def log(*a: object) -> None:
    print(*a, flush=True)


def draw_stratified(pool: list[tuple[str, str]], n: int, seed: int) -> list[dict]:
    """Per finding: up to ``n`` positive and as many negative patients, one film each.

    ``pool`` is ``(image, labels)``. Patients are visited in a seeded order; a patient's film is
    the first of theirs in that order that is eligible for the set.
    """
    order = np.random.default_rng(seed).permutation(len(pool))
    out = []
    for finding, lab in LABELS.items():
        sets: dict[bool, list[str]] = {True: [], False: []}
        seen: dict[bool, set[int]] = {True: set(), False: set()}
        for i in order:
            name, labels = pool[int(i)]
            pos = lab in labels
            if patient(name) in seen[pos]:
                continue
            seen[pos].add(patient(name))
            sets[pos].append(name)
        k = min(n, len(sets[True]))
        out += [{"finding": finding, "image": x, "label": 1} for x in sets[True][:k]]
        out += [{"finding": finding, "image": x, "label": 0} for x in sets[False][:k]]
    return out


def analyze(rows: list[dict], *, n_boot: int = 2000) -> dict:
    res: dict = {}
    diffs = []
    for finding in LABELS:
        rs = [r for r in rows if r["finding"] == finding]
        if not rs:
            continue
        y = [r["label"] for r in rs]
        sc = {k: [r[k] for r in rs] for k in SCORES}
        b = auc_bootstrap(y, sc, n_boot=n_boot)
        diffs += [b["diff"]["raw-s"], b["diff"]["norm-s"]]
        maj = [call(v) for v in sc["s"]]
        g = [r["greedy_parsed"] for r in rs]
        words = Counter((r["label"], w) for r in rs for w in r["samples_open"])
        tot = Counter()
        for (lab, _), c in words.items():
            tot[lab] += c
        res[finding] = {
            "n_pos": int(sum(y)), "n_neg": len(y) - int(sum(y)), **b,
            "calls_disagree_with_sampled": {k: int(sum(call(p) != m for p, m in
                                                       zip(sc[k], maj, strict=True)))
                                            for k in ("raw", "norm")},
            "calls_disagree_with_greedy": {k: int(sum(call(p) != a for p, a in
                                                      zip(sc[k], g, strict=True) if a != "NONE"))
                                           for k in ("raw", "norm")},
            "accuracy": {**{k: float(np.mean([(call(p) == "YES") == bool(t) for p, t in
                                              zip(sc[k], y, strict=True)]))
                            for k in ("raw", "norm", "s")},
                         "greedy": float(np.mean([a == ("YES" if t else "NO") for a, t in
                                                  zip(g, y, strict=True)]))},  # NONE is wrong
            "greedy_none": int(sum(a == "NONE" for a in g)),
            "mass_mean": float(np.mean([r["mass"] for r in rs])),
            "opening_by_label": {str(lab): dict([(w, c / tot[lab]) for (l1, w), c
                                                 in words.most_common() if l1 == lab][:5])
                                 for lab in (0, 1) if tot[lab]},
        }
    return {"by_finding": res, "verdict": eval_verdict(diffs)}


def compare(a: dict, b: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """E6-rank: paired film bootstrap of AUC_A - AUC_B under each score."""
    out: dict = {}
    flags = []
    for finding in LABELS:
        ka = {r["image"]: r for r in a["rows"] if r["finding"] == finding}
        kb = {r["image"]: r for r in b["rows"] if r["finding"] == finding}
        names = sorted(set(ka) & set(kb))
        if not names:
            continue
        y = np.array([ka[n]["label"] for n in names], dtype=float)
        A = {k: np.array([ka[n][k] for n in names]) for k in SCORES}
        B = {k: np.array([kb[n][k] for n in names]) for k in SCORES}
        rng = np.random.default_rng(seed)
        reps: dict[str, list[float]] = {k: [] for k in SCORES}
        for _ in range(n_boot):
            i = rng.integers(0, y.size, y.size)
            if y[i].sum() in (0, y.size):
                continue
            for k in SCORES:
                reps[k].append(auc(y[i], A[k][i]) - auc(y[i], B[k][i]))
        ci = {k: [auc(y, A[k]) - auc(y, B[k]), *map(float, np.quantile(reps[k], [0.025, 0.975]))]
              for k in SCORES}

        def side(c):  # noqa: ANN001, ANN202
            return 1 if c[1] > 0 else -1 if c[2] < 0 else 0

        s = side(ci["s"])
        for k in ("raw", "norm"):
            x = side(ci[k])
            flags.append("REVERSES" if x and s and x != s else "DIFFERS" if (x == 0) != (s == 0)
                         else "SAME")
        out[finding] = {"n": len(names), "auc_diff": ci}
    verdict = ("RANKING REVERSES" if "REVERSES" in flags else
               "RANKING DIFFERS IN SIGNIFICANCE" if "DIFFERS" in flags else "SAME RANKING")
    return {"by_finding": out, "verdict": verdict}


def report(res: dict) -> None:
    for f, r in res["by_finding"].items():
        log(f"\n  {f}: {r['n_pos']} pos / {r['n_neg']} neg, mass {r['mass_mean']:.3f}, "
            f"greedy NONE {r['greedy_none']}")
        for k in SCORES:
            lo, hi = r["ci"][k]
            log(f"    AUC {k:6s} {r['auc'][k]:.3f} [{lo:.3f}, {hi:.3f}]  "
                f"acc {r['accuracy'][k]:.3f}")
        for k in ("raw-s", "norm-s"):
            e, lo, hi = r["diff"][k]
            log(f"    {k:7s} {e:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        log(f"    calls vs sampled majority: {r['calls_disagree_with_sampled']}; "
            f"vs greedy: {r['calls_disagree_with_greedy']}")
    log(f"\n>>> E6: {res['verdict']}")


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="medgemma")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--analyze-only", type=Path, default=None)
    ap.add_argument("--compare", type=Path, nargs=2, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        a, b = (json.loads(p.read_text()) for p in args.compare)
        c = compare(a, b)
        for f, r in c["by_finding"].items():
            log(f"  {f} (n={r['n']}): " + "  ".join(
                f"{k} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for k, v in r["auc_diff"].items()))
        log(f">>> E6-rank ({a['model']} - {b['model']}): {c['verdict']}")
        return
    if args.analyze_only:
        report(analyze(json.loads(args.analyze_only.read_text())["rows"]))
        return

    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from tracecxr.core.config import MODELS as REGISTRY  # noqa: PLC0415
    from tracecxr.models._base import resolve_yes_no_token_ids, yes_probability  # noqa: PLC0415
    from transformers import __version__ as transformers_version  # noqa: PLC0415

    out = args.out or Path(f"/mnt/fast/clt/readout_eval_answers_{args.model}.json")
    img_dir = args.data_dir / "images"
    df = pd.read_csv(args.data_dir / "Data_Entry_2017.csv")
    have = {p.name for p in img_dir.glob("*.png") if not p.name.startswith("._")}
    df = df[df["Image Index"].isin(have)]
    pool = [(str(r["Image Index"]), str(r["Finding Labels"])) for _, r in df.iterrows()]
    films = draw_stratified(pool, args.n_per_class, args.seed)
    log(f"{len(films)} film x finding pairs: " + ", ".join(
        f"{f} {sum(x['label'] for x in films if x['finding'] == f)} pos"
        for f in LABELS))

    mid, rev = MODELS[args.model]
    mid = mid or REGISTRY["medgemma"].locator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "chexagent":  # custom code, transformers 4.40 (AMENDMENT 1)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(mid, revision=rev, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, trust_remote_code=True).to(torch.bfloat16).to(dev).eval()

        def encode(f: dict) -> dict:
            query = tok.from_list_format([{"image": str(img_dir / f["image"])},
                                          {"text": PROMPTS[f["finding"]]}])
            conv = [{"from": "system", "value": CHEXAGENT_SYSTEM},
                    {"from": "human", "value": query}]
            ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
            return {"input_ids": ids.to(dev), "use_cache": True}
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        proc = AutoProcessor.from_pretrained(mid, revision=rev)
        model = AutoModelForImageTextToText.from_pretrained(
            mid, revision=rev, dtype=torch.bfloat16).to(dev).eval()
        tok = getattr(proc, "tokenizer", proc)

        def encode(f: dict) -> dict:
            img = Image.open(img_dir / f["image"]).convert("RGB")
            return proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": PROMPTS[f["finding"]]}]}],
                add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt").to(dev)
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    log(f"loaded {mid}@{rev or 'registry'}; yes ids {yes_ids}, no ids {no_ids}")

    def readouts(row):
        row = row.float()
        pv = torch.softmax(row, -1)
        y, n = pv[yes_ids].max(), pv[no_ids].max()
        top = torch.topk(pv, 5)
        return {"raw": float(y), "norm": float(yes_probability(row, yes_ids, no_ids)),
                "mass": float(y + n), "gap": float(row[yes_ids].max() - row[no_ids].max()),
                "top5": [[tok.decode([int(i)]), float(p)]
                         for p, i in zip(top.values, top.indices, strict=True)]}

    prior = json.loads(out.read_text()) if out.exists() else {"rows": []}
    rows = list(prior["rows"])
    done = {(r["finding"], r["image"]) for r in rows}
    t0 = time.time()
    for k, f in enumerate(films):
        if (f["finding"], f["image"]) in done:
            continue
        torch.manual_seed(args.seed + k)
        inp = encode(f)
        plen = inp["input_ids"].shape[1]
        with torch.inference_mode():
            lg = model(input_ids=inp["input_ids"],
                       **{n: v for n, v in inp.items() if n not in ("input_ids", "use_cache")}
                       ).logits[0]
            scored = lg[-1]
            g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                               top_k=None, top_p=None, temperature=None,
                               repetition_penalty=1.0, output_logits=True,
                               return_dict_in_generate=True)
            s = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                               temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                               num_return_sequences=args.n_samples)
        ro, first = readouts(scored), readouts(g.logits[0][0])
        diff = max(abs(ro["raw"] - first["raw"]), abs(ro["norm"] - first["norm"]))
        if diff >= TOL:
            raise RuntimeError(f"{f['image']}: generation prefill differs ({diff:.4f})")
        greedy = tok.decode(g.sequences[0, plen:], skip_special_tokens=True)
        samples = tok.batch_decode(s[:, plen:], skip_special_tokens=True)
        parsed = [parse_answer(t) for t in samples]
        gp = parse_answer(greedy)
        rows.append({**f, **ro, "first_token_diff": diff, "greedy_text": greedy,
                     "greedy_parsed": gp, "greedy": GREEDY_SCORE[gp], "samples": samples,
                     "samples_parsed": parsed, "samples_open": [opening_word(t) for t in samples],
                     "s": yes_rate(parsed)})
        if (k + 1) % 25 == 0:
            log(f"  [{k + 1}/{len(films)} {time.time() - t0:.0f}s]")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"model": args.model, "rows": rows}, indent=1))

    res = analyze(rows)
    report(res)
    out.write_text(json.dumps(
        {"model": args.model, "model_id": mid, "revision": rev,
         "versions": {"torch": torch.__version__, "transformers": transformers_version},
         "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
         **res, "rows": rows}, indent=1))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
