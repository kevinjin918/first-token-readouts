"""Fair three-way eval: score every CLT on the SAME diverse held-out set.

The cached models were scored on their own narrow held-out (~6k-image distribution); the
model on a diverse one. That's apples-to-oranges. This captures ONE common diverse held-out (tail
the corpus, disjoint from training) and reconstructs it with each checkpoint, so the FVU table is
finally comparable. The hypothesis: the diverse-trained model holds, the narrow-trained ones drop.

Pass models as repeated --model name:ckpt:result triples. GPU.

Usage: python scripts/clt_fair_eval.py \
  --model onthefly:/mnt/clt-cache/ckpt_live/clt_ckpt.pt:~/clt_live_result.json \
  --model cached_full:/mnt/clt-cache/ckpt/clt_ckpt.pt:~/clt_scale_result.json \
  --model per_layer:/mnt/clt-cache/ckpt_span0/clt_ckpt.pt:~/clt_span0_result.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tracecxr.transcoder import eval as clt_eval
from tracecxr.transcoder.cache import medgemma_capture_fn
from tracecxr.transcoder.clt import CLTConfig, CrossLayerTranscoder
from tracecxr.transcoder.stream import capture_stack

LAYERS = list(range(34))
CXR_GLOB = str(Path.home() / ".cache/tracecxr/data/chestxray14/images/*.png")
CSV = str(Path.home() / ".cache/tracecxr/data/chexpert_plus/df_chexpert_plus_240401.csv")


def log(*a: object) -> None:
    print(*a, flush=True)


def loading_capture(layers: list[int]):
    base = medgemma_capture_fn(layers)

    def cap(rec):
        img = None
        if rec.image_path is not None:
            from PIL import Image  # noqa: PLC0415

            img = Image.open(rec.image_path).convert("RGB")
        return base(SimpleNamespace(id=rec.id, text=rec.text, image=img))

    return cap


def fvu_of(ckpt: str, result: str, x_te, y_te, k_te, n_layers: int) -> dict[str, float]:
    import torch  # noqa: PLC0415

    res = json.loads(Path(result).read_text()) if Path(result).exists() else {}
    cfg = CLTConfig(n_features=res.get("n_features", 2048), span=res.get("span"),
                    k=res.get("k", 32), n_layers=n_layers, amp=True,
                    adam_8bit=bool(res.get("adam_8bit", False)))
    clt = CrossLayerTranscoder(cfg)
    clt.load_checkpoint(ckpt, load_optimizer=False)
    dev = clt._device()
    act_dtype = torch.bfloat16 if (cfg.amp and dev == "cuda") else torch.float32
    parts = []
    with torch.inference_mode():
        for i in range(0, x_te.shape[0], 4096):
            xb = torch.as_tensor(x_te[i : i + 4096], dtype=act_dtype, device=dev)
            with clt._autocast():
                parts.append(clt.reconstruct(xb).float().cpu().numpy())
    recon = np.concatenate(parts, 0)
    pl = [clt_eval.fvu_by_kind(y_te[:, li], recon[:, li], k_te) for li in range(n_layers)]
    out = {f"fvu_{key}": float(np.nanmean([p[key] for p in pl])) for key in pl[0].keys()}
    import gc  # noqa: PLC0415

    del clt
    gc.collect()
    torch.cuda.empty_cache()  # fully free before the next model loads (3 x ~25 GB sequentially)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, help="name:ckpt:result (repeatable)")
    ap.add_argument("--n-img", type=int, default=150)
    ap.add_argument("--n-text", type=int, default=80)
    ap.add_argument("--out", default=str(Path.home() / "clt_fair_eval.json"))
    args = ap.parse_args()

    # common diverse held-out: tail of the corpus (disjoint from the training head all models used)
    import pandas as pd  # noqa: PLC0415

    reports = [r for r in pd.read_csv(CSV)["report"].tolist() if isinstance(r, str) and len(r) > 40]
    recs = [SimpleNamespace(id=f"t{i}", text=r[:1500], image_path=None)
            for i, r in enumerate(reports[-args.n_text:])]
    recs += [SimpleNamespace(id=f"i{i}", text="Interpret this chest X-ray.", image_path=p)
             for i, p in enumerate(sorted(glob.glob(CXR_GLOB))[-args.n_img:])]
    random.Random(0).shuffle(recs)
    log(f"common held-out: {len(recs)} records; capturing once...")
    x_te, y_te, k_te = capture_stack(recs, loading_capture(LAYERS), LAYERS)

    rows = []
    for spec in args.model:
        name, ckpt, result = spec.split(":", 2)
        ckpt, result = str(Path(ckpt).expanduser()), str(Path(result).expanduser())
        sc = fvu_of(ckpt, result, x_te, y_te, k_te, len(LAYERS))
        rows.append({"name": name, **sc})
        log(f"  {name:14s}  fvu_image {sc.get('fvu_image', float('nan')):.3f}  "
            f"fvu_text {sc.get('fvu_text', float('nan')):.3f}  "
            f"fvu_all {sc.get('fvu_all', float('nan')):.3f}")
    Path(args.out).write_text(json.dumps({"n_records": len(recs), "models": rows}, indent=2))
    log(f"saved {args.out}")
    log("read: lower fvu on this COMMON diverse set = the better representation.")


if __name__ == "__main__":
    main()
