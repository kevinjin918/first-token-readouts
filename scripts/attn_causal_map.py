"""Does a patch's attention weight predict its causal effect? And is the AP bias a severity prior?

Two experiments, both designed to be falsifiable before running.

A) ATTENTION-VS-CAUSE, per patch. For each film, occlude one 2x2 block of the 16x16 patch grid at
   a time (64 blocks) and record the change in P(cardiomegaly). Correlate each block's mean
   attention weight against its own causal effect, within film, by Spearman rho.

   PRE-SPECIFIED: if the attention map is a faithful explanation of what the model used, rho
   should be substantially POSITIVE (we set >= +0.3 as "attention explains"). rho near zero means
   attention carries no information about causal relevance. rho negative means it is actively
   misleading. This supersedes the coarser border-vs-random comparison in shortcut_probe.py, which
   could be dismissed as "the heart is in the middle": here every block is compared against its own
   attention weight, so anatomy is not confounded with the test.

   CONTROL: the same correlation computed against a shuffled attention map, which must come out at
   rho ~ 0. This catches any artifact of the occlusion procedure itself producing structure.

B) SEVERITY-PRIOR MEDIATION. Restricting the negative set to NIH "No Finding" collapses the AP/PA
   cardiomegaly gap from +0.303 to +0.052, which suggests the model raises P(cardiomegaly) on films
   that merely look sick. Direct test: within cardiomegaly-NEGATIVE AP films, regress
   P(cardiomegaly) on the number of OTHER findings the film carries.

   PRE-SPECIFIED: a positive monotone relationship supports the severity-prior account. A flat one
   refutes it and would mean the "No Finding" result came from something else about that subset.

GPU. Usage (VM):
  python scripts/attn_causal_map.py --n-films 20 --n-severity 200 \
    --data-dir ~/.cache/.../chestxray14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path.home() / ".cache/tracecxr/data/chestxray14"
PROMPT = "Does this chest X-ray show cardiomegaly? Answer yes or no."
GRID, BLOCK = 16, 2          # 16x16 patch grid, occluded in 2x2 blocks -> 64 blocks
FIRE = 0.5

ATTN: dict[int, object] = {}
ORDER: list[int] = []


def log(*a: object) -> None:
    print(*a, flush=True)


def install_capture():
    import transformers.models.gemma3.modeling_gemma3 as gm  # noqa: PLC0415

    orig = gm.eager_attention_forward

    def patched(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                softcap=None, **kw):
        out, w = orig(module, query, key, value, attention_mask,
                      dropout=dropout, scaling=scaling, softcap=softcap, **kw)
        if id(module) not in ORDER:
            ORDER.append(id(module))
        ATTN[id(module)] = w.detach()
        return out, w

    gm.eager_attention_forward = patched


def occlude_block(arr: np.ndarray, br: int, bc: int) -> Image.Image:
    """Black out one BLOCKxBLOCK block of the GRIDxGRID cell grid."""
    a = arr.copy()
    h, w = a.shape[:2]
    r0, r1 = br * BLOCK, (br + 1) * BLOCK
    c0, c1 = bc * BLOCK, (bc + 1) * BLOCK
    a[int(r0 * h / GRID):int(r1 * h / GRID), int(c0 * w / GRID):int(c1 * w / GRID)] = 0
    return Image.fromarray(a)


def load_rows(data_dir: Path):
    import pandas as pd  # noqa: PLC0415
    return pd.read_csv(data_dir / "Data_Entry_2017.csv")


def main() -> None:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

    try:
        from tracecxr.attribution.medgemma_backend import (  # noqa: PLC0415
            best_answer_token_id,
        )
        from tracecxr.core.config import MODELS  # noqa: PLC0415
        locator = MODELS["medgemma"].locator
    except ImportError:
        locator = "google/medgemma-1.5-4b-it"

        def best_answer_token_id(tok, answer, scores):  # noqa: PLC0415
            best_id, best, seen = None, -float("inf"), set()
            for v in {answer, answer.lower(), answer.upper(), answer.capitalize(),
                      " " + answer, " " + answer.lower(), " " + answer.capitalize()}:
                enc = tok.encode(v, add_special_tokens=False)
                if not enc or int(enc[0]) in seen:
                    continue
                seen.add(int(enc[0]))
                if float(scores[int(enc[0])]) > best:
                    best, best_id = float(scores[int(enc[0])]), int(enc[0])
            return best_id

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-films", type=int, default=20, help="films for experiment A")
    ap.add_argument("--n-severity", type=int, default=200, help="films for experiment B")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", default="/mnt/big/attnpatch/attn_causal_map.json")
    args = ap.parse_args()

    from scipy.stats import spearmanr  # noqa: PLC0415

    install_capture()
    proc = AutoProcessor.from_pretrained(locator)
    model = AutoModelForImageTextToText.from_pretrained(
        locator, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tok = getattr(proc, "tokenizer", proc)
    img_tok = int(model.config.image_token_index)

    def run(image):
        inp = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": PROMPT}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to("cuda")
        d = inp["input_ids"].shape[1] - 1
        with torch.no_grad():
            lg = model(**inp).logits[0, d].float()
        pv = torch.softmax(lg, -1).cpu().numpy()
        return float(pv[best_answer_token_id(tok, "yes", pv)]), inp, d

    def grid_attn(inp, d):
        ids = inp["input_ids"][0]
        cols = (ids == img_tok).nonzero(as_tuple=True)[0]
        row = torch.stack([ATTN[m][0, :, d, :].float().mean(0) for m in ORDER]).mean(0)
        g = row[cols].cpu().numpy()
        return g.reshape(GRID, GRID)

    df = load_rows(args.data_dir)
    img_dir = args.data_dir / "images"
    rng = np.random.default_rng(args.seed)

    def eligible(view):
        return [(str(img_dir / str(r["Image Index"])), str(r["Finding Labels"]))
                for _, r in df.iterrows()
                if not str(r["Image Index"]).startswith("._")
                and "Cardiomegaly" not in str(r["Finding Labels"])
                and str(r["View Position"]) == view
                and (img_dir / str(r["Image Index"])).exists()]

    ap_pool = eligible("AP")

    # ---------- A: per-patch attention vs causal effect ------------------------------
    log("=== A: attention weight vs causal effect, per patch ===")
    nb = GRID // BLOCK
    idx = rng.permutation(len(ap_pool))
    recs = []
    for i in idx:
        if len(recs) >= args.n_films:
            break
        path, _ = ap_pool[i]
        arr = np.array(Image.open(path).convert("RGB"))
        base, inp, d = run(Image.open(path).convert("RGB"))
        if base < FIRE:
            continue
        att = grid_attn(inp, d)
        block_att, block_eff = [], []
        for br in range(nb):
            for bc in range(nb):
                block_att.append(float(att[br * BLOCK:(br + 1) * BLOCK,
                                            bc * BLOCK:(bc + 1) * BLOCK].sum()))
                block_eff.append(base - run(occlude_block(arr, br, bc))[0])
        rho, p = spearmanr(block_att, block_eff)
        sh = rng.permutation(block_att)          # control: shuffled attention map
        rho_sh, _ = spearmanr(sh, block_eff)
        recs.append({"image": Path(path).name, "base": base,
                     "rho": float(rho), "p": float(p), "rho_shuffled": float(rho_sh),
                     "block_attn": block_att, "block_effect": block_eff})
        log(f"  {Path(path).name}: base {base:.3f}  rho {rho:+.3f} (p={p:.3f})  "
            f"shuffled {rho_sh:+.3f}")

    rhos = np.array([r["rho"] for r in recs])
    rhos_sh = np.array([r["rho_shuffled"] for r in recs])
    log(f"\n  n={len(recs)} films")
    log(f"  mean Spearman rho (attention vs causal effect) = {rhos.mean():+.3f} "
        f"(sd {rhos.std(ddof=1):.3f}, {int((rhos > 0).sum())}/{len(rhos)} positive)")
    log(f"  mean rho with SHUFFLED attention (control)     = {rhos_sh.mean():+.3f}")
    verdict_a = ("attention explains" if rhos.mean() >= 0.3 else
                 "attention is actively misleading" if rhos.mean() <= -0.1 else
                 "attention carries little information about causal relevance")
    log(f">>> {verdict_a}")

    # ---------- B: severity prior ------------------------------------------------------
    log("\n=== B: does P(cardiomegaly) rise with other findings, on AP negatives? ===")
    sev = []
    for i in rng.permutation(len(ap_pool))[:args.n_severity]:
        path, labels = ap_pool[i]
        k = 0 if labels == "No Finding" else len([x for x in labels.split("|") if x])
        sev.append({"image": Path(path).name, "n_findings": k,
                    "p": run(Image.open(path).convert("RGB"))[0]})
    ks = np.array([s["n_findings"] for s in sev])
    ps = np.array([s["p"] for s in sev])
    rho_b, p_b = spearmanr(ks, ps)
    log(f"  n={len(sev)} AP cardiomegaly-negative films")
    for k in sorted(set(ks.tolist())):
        m = ks == k
        if m.sum() >= 3:
            log(f"    {k} other finding(s): n={int(m.sum())}  mean P = {ps[m].mean():.3f}")
    log(f"  Spearman rho(n_findings, P) = {rho_b:+.3f}, p = {p_b:.5f}")
    verdict_b = ("severity prior supported" if rho_b > 0 and p_b < 0.05
                 else "no monotone severity relationship")
    log(f">>> {verdict_b}")

    Path(args.out).write_text(json.dumps({
        "A": {"n_films": len(recs), "mean_rho": float(rhos.mean()),
              "sd_rho": float(rhos.std(ddof=1)) if len(rhos) > 1 else None,
              "n_positive": int((rhos > 0).sum()),
              "mean_rho_shuffled": float(rhos_sh.mean()),
              "criterion": "rho >= +0.3 => attention explains; <= -0.1 => misleading",
              "verdict": verdict_a, "records": recs},
        "B": {"n": len(sev), "spearman_rho": float(rho_b), "p": float(p_b),
              "verdict": verdict_b, "records": sev},
    }, indent=2))
    log(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
