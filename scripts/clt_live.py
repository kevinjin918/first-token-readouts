"""Train the max CLT ON-THE-FLY — no disk cache, data bounded only by the image/report corpus.

The cached run (clt_scale.py) caps at the fast disk (~2 M tokens), which **undertrained** the 6.24 B
cross-layer decoder — the per-layer baseline beat it (results/clt_scale.md). This trades the cache
for a live MedGemma forward each buffer (only ~6-12% of a big-CLT step) so the model sees *all* the
data: every NIH/VinDr/CheXpert image + every report, cycled. Coverage ~= (records_per_buffer /
steps_per_buffer) x steps records; raise --steps for more unique data (no disk limit).

MedGemma is resident **alongside** the CLT (~8 GB on top of the CLT's training footprint), so the
default batch is 768 (vs 1024 cached) for headroom on 80 GB. Logs incrementally + optional --wandb.

Usage (VM): python scripts/clt_live.py --n-img 50000 --steps 50000 --wandb
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from types import SimpleNamespace

from tracecxr.transcoder.cache import medgemma_capture_fn
from tracecxr.transcoder.clt import CLTConfig
from tracecxr.transcoder.train import train_clt_onthefly

LAYERS = list(range(34))
CKPT = Path.home() / "clt_live_ckpt"
OUT = Path.home() / "clt_live_result.json"
CXR_GLOB = str(Path.home() / ".cache/tracecxr/data/chestxray14/images/*.png")
CSV = str(Path.home() / ".cache/tracecxr/data/chexpert_plus/df_chexpert_plus_240401.csv")
SEED = 0


def log(*a: object) -> None:
    print(*a, flush=True)


# The image prompt distribution the dictionary is trained on. The original run used ONLY the
# first entry, which made the published faithfulness number (top-1 1.00 / KL 0.044) an
# in-distribution measurement: on any other prompt the substituted model degrades badly, and on the
# constrained yes/no probes that every mechanistic claim in this project uses it collapses entirely
# (0.00 agreement, KL 13.5). See results/clt_scale/metric_transfer.md. Training over the probe
# distribution is what makes feature-level conclusions on those probes meaningful.
IMG_PROMPTS_OPEN = ["Interpret this chest X-ray."]
IMG_PROMPTS_PROBE = [
    "Does this chest X-ray show cardiomegaly? Answer yes or no.",
    "Does this chest X-ray show a pleural effusion? Answer yes or no.",
    "Does this chest X-ray show a pneumothorax? Answer yes or no.",
    "Does this chest X-ray show consolidation? Answer yes or no.",
    "Does this chest X-ray show pulmonary edema? Answer yes or no.",
    "Is there cardiomegaly (an enlarged heart)? Answer yes or no.",
    "Describe in detail all findings visible in this chest radiograph.",
]
PROMPT_SETS = {
    "open": IMG_PROMPTS_OPEN,                              # the original, reproduces the old run
    "probe": IMG_PROMPTS_PROBE,                            # constrained probes only
    "mixed": IMG_PROMPTS_OPEN * 2 + IMG_PROMPTS_PROBE,     # both, open-ended kept in proportion
}


def build_records(n_text: int, n_img: int, prompt_set: str = "open") -> list[SimpleNamespace]:
    """Lightweight records (images by path, opened lazily), shuffled to mix kinds across buffers.

    prompt_set selects the image prompt distribution; images are cycled over it so the dictionary
    sees each prompt in equal proportion.
    """
    import pandas as pd  # noqa: PLC0415

    prompts = PROMPT_SETS[prompt_set]
    # Only touch the report CSV when text records are actually requested: an image-only run is
    # valid (the constrained probes carry their own prompt text) and should not require it.
    text = []
    if n_text > 0:
        reports = [r for r in pd.read_csv(CSV)["report"].tolist()
                   if isinstance(r, str) and len(r) > 40]
        text = [SimpleNamespace(id=f"t{i}", text=r[:1500], image_path=None)
                for i, r in enumerate(reports[:n_text])]
    imgs = [SimpleNamespace(id=f"i{i}", text=prompts[i % len(prompts)], image_path=p)
            for i, p in enumerate(sorted(glob.glob(CXR_GLOB))[:n_img])]
    recs = text + imgs
    random.Random(SEED).shuffle(recs)
    return recs


def loading_capture(layers: list[int]):
    """Wrap the MedGemma capture so records carry image paths, opened lazily per record."""
    base = medgemma_capture_fn(layers)

    def cap(rec: SimpleNamespace):
        img = None
        if rec.image_path is not None:
            from PIL import Image  # noqa: PLC0415

            img = Image.open(rec.image_path).convert("RGB")
        return base(SimpleNamespace(id=rec.id, text=rec.text, image=img))

    return cap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=int, default=2048)
    p.add_argument("--span", type=int, default=-1, help="-1 = full cross-layer; 0 = per-layer")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--prompt-set", choices=tuple(PROMPT_SETS), default="open",
                   help="image prompt distribution to train on; 'mixed' covers the yes/no probes "
                        "that mechanistic claims are measured with (see metric_transfer.md)")
    p.add_argument("--batch", type=int, default=768, help="< cached: MedGemma is resident too")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-img", type=int, default=50000, help="CXR images in the training pool")
    p.add_argument("--n-text", type=int, default=10000, help="reports in the training pool")
    p.add_argument("--n-eval", type=int, default=80, help="held-out images+reports for FVU")
    p.add_argument("--records-per-buffer", type=int, default=96)
    p.add_argument("--steps-per-buffer", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--checkpoint-every", type=int, default=2500)
    p.add_argument("--no-adam-8bit", dest="adam_8bit", action="store_false")
    p.add_argument("--heartbeat-every", type=int, default=200)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="tracecxr-clt")
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    span_cfg = None if args.span < 0 else args.span
    span_eff = (len(LAYERS) - 1) if span_cfg is None else span_cfg
    span_label = f"full({span_eff})" if span_cfg is None else str(span_eff)

    recs = build_records(args.n_text, args.n_img, args.prompt_set)
    eval_recs, train_recs = recs[: args.n_eval], recs[args.n_eval :]
    n_img = sum(1 for r in train_recs if r.image_path)
    log(f"corpus: {len(train_recs)} train ({n_img} img + {len(train_recs)-n_img} text) + "
        f"{len(eval_recs)} held-out; span {span_label}, features {args.features}")
    coverage = args.records_per_buffer / args.steps_per_buffer * args.steps
    log(f"~{int(coverage)} records will be captured over {args.steps} steps "
        f"(vs 8500 in the cached run) -> live data, no disk cap")

    cfg = CLTConfig(n_features=args.features, k=args.k, span=span_cfg,
                    learning_rate=args.lr, amp=True, adam_8bit=args.adam_8bit)
    cap = loading_capture(LAYERS)

    run_meta = {"features": args.features, "span": span_eff, "k": args.k, "batch": args.batch,
                "steps": args.steps, "mode": "on-the-fly", "n_img_pool": args.n_img,
                "prompt_set": args.prompt_set, "img_prompts": PROMPT_SETS[args.prompt_set]}
    wb = None
    if args.wandb:
        import wandb  # noqa: PLC0415
        wb = wandb.init(project=args.wandb_project, config=run_meta)
    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    metrics_f = open(Path(args.ckpt) / "metrics.jsonl", "a")  # noqa: SIM115

    def progress(rec: dict) -> None:
        if "fvu_image" in rec:
            log(f"  step {rec['step']}  recon_mse {rec.get('recon_mse', 0):.4f}  "
                f"fvu_text {rec.get('fvu_text', float('nan')):.3f}  "
                f"fvu_image {rec.get('fvu_image', float('nan')):.3f}  l0 {rec.get('l0', 0):.1f}")
        else:
            log(f"  [hb] step {rec['step']}  recon_mse {rec.get('recon_mse', 0):.4f}")
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        if wb is not None:
            wb.log(rec, step=rec["step"])

    log(f"training on-the-fly: {args.steps} steps, batch {args.batch}, capture "
        f"{args.records_per_buffer} recs / {args.steps_per_buffer} steps; ckpt {args.ckpt}")
    out = train_clt_onthefly(
        train_recs, cap, eval_recs, LAYERS, cfg, steps=args.steps, batch_size=args.batch,
        eval_every=args.eval_every, records_per_buffer=args.records_per_buffer,
        steps_per_buffer=args.steps_per_buffer, checkpoint_dir=args.ckpt,
        checkpoint_every=args.checkpoint_every, progress_fn=progress,
        heartbeat_every=args.heartbeat_every,
    )
    metrics_f.close()
    if wb is not None:
        wb.finish()
    log("FINAL scorecard:", {k: round(v, 4) for k, v in out["scorecard"].items()})
    Path(args.out).write_text(json.dumps({
        "mode": "on-the-fly", "layers": LAYERS, "n_features": args.features, "span": span_eff,
        "k": args.k, "steps": args.steps, "n_img_pool": args.n_img,
        "prompt_set": args.prompt_set, "img_prompts": PROMPT_SETS[args.prompt_set],
        "scorecard": out["scorecard"], "history": out["history"],
    }, indent=2))
    log(f"saved {args.out}; CLT checkpoint at {args.ckpt}/clt_ckpt.pt")
    log("=== DONE clt_live ===")


if __name__ == "__main__":
    main()
