"""Diagnostic: does the published CLT faithfulness metric transfer to the decision we study?

Our evidence-capture test (clt_carries_evidence.py) came back with the substituted model emitting
P(yes) ~1e-6 on every input, which VOIDS that test: its pre-specified validity precondition was
that the substituted model must first reproduce the clean behaviour, and it does not. Before any
claim is made, we have to separate two explanations:

  (A) our substitution setup is broken (wrong config, layer order, dtype), or
  (B) the substitution is behaving exactly as published, and the published metric simply does not
      transfer to a constrained yes/no read-out.

The banked number (results/clt_scale/clt_replacement*.json: top-1 agreement 1.00, KL 0.044 on image
tokens) was measured with an OPEN-ENDED prompt, "Interpret this chest X-ray.", scoring the
next-token distribution. Every mechanistic claim in this project instead rests on a constrained
yes/no probe. Those are not the same measurement.

TEST. Same images, same substitution, two prompts, three numbers each (top-1 agreement, KL, and
P(yes) ratio where applicable):

  open-ended  "Interpret this chest X-ray."                 -> should reproduce ~1.00 / ~0.044
  constrained "Does this chest X-ray show cardiomegaly? ..." -> measured

PRE-SPECIFIED:
  open-ended agreement >= 0.9 and KL <= 0.15  => setup is sound, explanation (A) is out. If the
      constrained prompt then collapses, that is a real and reportable result: the field's
      replacement-faithfulness metric is insensitive to the behaviour mechanistic claims are made
      about.
  open-ended agreement < 0.9                  => our setup is broken. Report that, fix it, and make
      no claim about the dictionary.

GPU. Usage (VM):
  python scripts/clt_metric_transfer.py --ckpt /mnt/big/ckpt_live/clt_ckpt.pt \
    --result ~/clt_live_result.json --n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# The battery discriminates "constrained FORMAT" from "this particular prompt". If the trivial
# constrained question (is this even a chest X-ray?) also collapses, the substitution is destroying
# format/instruction control, not diagnostic content. If only the diagnostic ones collapse, it is
# content-specific. Two open-ended prompts of different lengths rule out a pure length effect.
PROMPTS = [
    ("open_published", "Interpret this chest X-ray."),
    ("open_long", "Describe in detail all findings visible in this chest radiograph, "
                  "including the cardiomediastinal silhouette, lungs, pleura and bones."),
    ("constr_cardio", "Does this chest X-ray show cardiomegaly? Answer yes or no."),
    ("constr_effusion", "Does this chest X-ray show a pleural effusion? Answer yes or no."),
    ("constr_trivial", "Is this image a chest X-ray? Answer yes or no."),
    ("constr_nonmedical", "Is this image in colour? Answer yes or no."),
]
OPEN, CONSTRAINED = PROMPTS[0][1], PROMPTS[2][1]


def log(*a: object) -> None:
    print(*a, flush=True)


def main() -> None:
    import torch  # noqa: PLC0415
    from clt_replacement import replacement_logits  # noqa: PLC0415
    from tracecxr.attribution.medgemma_backend import (  # noqa: PLC0415
        MedGemmaCLTBackend,
        best_answer_token_id,
    )
    from tracecxr.transcoder.clt import CLTConfig  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="/mnt/big/ckpt_live/clt_ckpt.pt")
    ap.add_argument("--result", default=str(Path.home() / "clt_live_result.json"))
    ap.add_argument("--banked", default="/mnt/big/attnpatch/attn_causal_map.json")
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".cache/tracecxr/data/chestxray14")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="/mnt/big/attnpatch/clt_metric_transfer.json")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    layers = list(res["layers"])
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span", 33),
                    k=res.get("k", 32), n_layers=len(layers), amp=True,
                    adam_8bit=bool(res.get("adam_8bit", False)))
    backend = MedGemmaCLTBackend(args.ckpt, layers, clt_config=cfg)
    backend._load()
    state = backend._state
    proc, model = state["proc"], state["model"]
    tok = getattr(proc, "tokenizer", proc)
    log(f"loaded MedGemma + CLT ({cfg.n_features} feats, span {cfg.span}, {len(layers)} layers)")

    def build(image, prompt):
        return proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(model.device)

    banked = json.loads(Path(args.banked).read_text())["A"]["records"][: args.n]
    img_dir = args.data_dir / "images"
    out: dict[str, list] = {name: [] for name, _ in PROMPTS}

    for b in banked:
        path = img_dir / b["image"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        for name, prompt in PROMPTS:
            o, r = replacement_logits(state, layers, build(img, prompt))
            po = torch.softmax(o, -1)
            pr = torch.softmax(r, -1)
            kl = float((po * (po.clamp_min(1e-12).log() - pr.clamp_min(1e-12).log())).sum())
            agree = bool(int(o.argmax()) == int(r.argmax()))
            rec = {"image": b["image"], "kl": kl, "agree": agree,
                   "top_orig": tok.decode([int(o.argmax())]),
                   "top_repl": tok.decode([int(r.argmax())])}
            if name.startswith("constr"):
                pv_o, pv_r = po.cpu().numpy(), pr.cpu().numpy()
                rec["p_yes_orig"] = float(pv_o[best_answer_token_id(tok, "yes", pv_o)])
                rec["p_yes_repl"] = float(pv_r[best_answer_token_id(tok, "yes", pv_r)])
            out[name].append(rec)
        log("  " + b["image"] + "  " + "  ".join(
            f"{n}: agr={out[n][-1]['agree']:d} KL={out[n][-1]['kl']:.2f}" for n, _ in PROMPTS))

    log(f"\n=== n={len(out[PROMPTS[0][0]])} images ===")
    log(f"  {'prompt':18s} {'top-1 agree':>11s} {'mean KL':>9s} {'P(ans) orig->repl':>22s}")
    for name, _ in PROMPTS:
        rs = out[name]
        ag = float(np.mean([r["agree"] for r in rs]))
        kl = float(np.mean([r["kl"] for r in rs]))
        pa = ""
        if "p_yes_orig" in rs[0]:
            pa = (f"{np.mean([r['p_yes_orig'] for r in rs]):.3f} -> "
                  f"{np.mean([r['p_yes_repl'] for r in rs]):.1e}")
        log(f"  {name:18s} {ag:>11.2f} {kl:>9.3f} {pa:>22s}")

    ag_open = float(np.mean([r["agree"] for r in out["open_published"]]))
    kl_open = float(np.mean([r["kl"] for r in out["open_published"]]))
    if ag_open >= 0.9 and kl_open <= 0.15:
        verdict = ("SETUP SOUND: reproduces the published open-ended metric. The collapse on the "
                   "constrained probe is therefore a real property, not a bug.")
    else:
        verdict = ("SETUP SUSPECT: does not reproduce the published open-ended metric either. "
                   "Make no claim about the dictionary until this is fixed.")
    log(f">>> {verdict}")

    Path(args.out).write_text(json.dumps({"verdict": verdict, "results": out}, indent=2))
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
