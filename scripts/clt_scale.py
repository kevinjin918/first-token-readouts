"""Train the max single-H100 cross-layer transcoder for MedGemma (docs/clt_spec.md).

The release artifact: ~6.24 B params, all 34 layers, FULL cross-layer span (every feature writes to
all downstream layers), 2048 features, 8-bit Adam — trained by **streaming** the merged activation
cache (scripts/clt_cache.py) through a bounded shuffle buffer with checkpoint/resume (spot-safe).
MedGemma is NOT resident here — we train on cached activations, so the full 80 GB is the
transcoder's.

Default config is the H100-verified max (~64 GB peak at batch 1024; see docs/clt_spec.md). The
W_dec split (clt.py) lifted the 8-bit-kernel span cap, so memory is the only ceiling — drop
--features or --batch if you OOM. Trade span<->features at this ~6 B budget via --span / --features.

Usage (after caching + merge):
  python scripts/clt_scale.py                        # full cross-layer, 2048 feats (the max)
  python scripts/clt_scale.py --span 8 --features 4096   # less span, wider dictionary (~3.6 B)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tracecxr.transcoder.cache import load_manifest
from tracecxr.transcoder.clt import CLTConfig
from tracecxr.transcoder.train import train_clt_streaming

CACHE = Path.home() / "clt_scale_cache"
CKPT = Path.home() / "clt_scale_ckpt"
OUT = Path.home() / "clt_scale_result.json"


def log(*a: object) -> None:
    print(*a, flush=True)


def estimate_params(n_layers: int, span: int, n_features: int, d_model: int) -> tuple[int, int]:
    """(encoder, decoder) param counts. Decoder writes to span+1 layers per source layer."""
    enc = n_layers * d_model * n_features
    dec = n_layers * (span + 1) * n_features * d_model
    return enc, dec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Defaults are the H100-verified MAX-cross-layer config (see docs/clt_spec.md): 2048 feats /
    # FULL span (every feature writes to all 34 downstream layers) = 6.24 B params, ~64 GB peak at
    # batch 1024. The W_dec split (clt.py) removed the 8-bit-kernel span cap, so memory is the only
    # ceiling: 2560 feats (7.8 B) is too tight (79 GB). Trade span<->features at this ~6 B budget.
    p.add_argument("--features", type=int, default=2048, help="features per layer")
    p.add_argument("--span", type=int, default=-1,
                   help="cross-layer write span; -1 = full (all downstream layers), 0 = per-layer")
    p.add_argument("--k", type=int, default=32, help="TopK active features per token")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--checkpoint-every", type=int, default=2500)
    p.add_argument("--buffer-shards", type=int, default=8, help="shards resident in RAM (mem knob)")
    p.add_argument("--test-frac", type=float, default=0.1, help="fraction of shards held out")
    p.add_argument("--max-test-tokens", type=int, default=20000,
                   help="cap held-out tokens (else a 10%% split of millions OOMs at start)")
    p.add_argument("--no-adam-8bit", dest="adam_8bit", action="store_false",
                   help="use fp32 Adam (no kernel limit, but ~2x optimizer memory)")
    p.add_argument("--heartbeat-every", type=int, default=200,
                   help="log a step/loss heartbeat every N steps (live progress, no eval)")
    p.add_argument("--wandb", action="store_true",
                   help="log metrics to Weights & Biases for a live dashboard (needs wandb login)")
    p.add_argument("--wandb-project", default="tracecxr-clt")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    man = load_manifest(args.cache)
    layers = list(man["layers"])
    n_layers = len(layers)
    # Confirm d_model from a real cached shard (don't assume 2560).
    first = man["shards"][0]
    with np.load(Path(args.cache) / first) as z:
        d_model = int(z[f"in_{layers[0]}"].shape[-1])
    tokens = man.get("token_counts", {})
    span_cfg = None if args.span < 0 else args.span  # None -> full cross-layer in CLTConfig
    span_eff = (n_layers - 1) if span_cfg is None else min(span_cfg, n_layers - 1)  # for param math
    span_label = f"full({span_eff})" if span_cfg is None else str(span_eff)
    enc, dec = estimate_params(n_layers, span_eff, args.features, d_model)
    total = enc + dec
    log(f"cache: {len(man['shards'])} shards, layers {layers[0]}-{layers[-1]} ({n_layers}), "
        f"d_model {d_model}, tokens {tokens}")
    log(f"CLT: features={args.features}, span={span_label}, k={args.k}, adam_8bit={args.adam_8bit}")
    log(f"  params ~ {total/1e9:.2f} B  (encoder {enc/1e9:.2f} B + decoder {dec/1e9:.2f} B)")
    # Params are fp32 (amp casts only matmul compute, not storage): weights 4 + grads 4 + Adam
    # states (8-bit 2 / fp32 8) B/param. Activation graph adds ~5-15 GB by batch (verified default
    # full/2048 peaks ~64 GB on the H100). NOT bf16 weights -- that mistake is why ~8.6 B OOMs.
    opt_bytes = 2 if args.adam_8bit else 8
    mem_gb = total * (4 + 4 + opt_bytes) / 1e9
    log(f"  est. fixed GPU mem (weights+grads+opt) ~ {mem_gb:.0f} GB + ~10-25 GB activation graph")
    # 8-bit Adam's CUDA kernel uses int32 indexing -> each param tensor must stay < 2^31 elems.
    # W_dec is split per write-offset (clt.py), so the largest decoder tensor is n_layers*features*
    # d_model (independent of span) -- span no longer bounded by the kernel, only features.
    wdec_tensor = n_layers * args.features * d_model
    if args.adam_8bit and wdec_tensor >= 2_147_000_000:
        raise SystemExit(
            f"per-offset W_dec is {wdec_tensor/1e9:.2f} B elements >= the 8-bit Adam kernel limit "
            f"(2.147 B). It will fail with 'invalid argument ... ops.cu'. Lower --features below "
            f"{2_147_000_000//(n_layers*d_model)}, or pass --no-adam-8bit for fp32 Adam.")

    cfg = CLTConfig(n_features=args.features, k=args.k, span=span_cfg,
                    learning_rate=args.lr, amp=True, adam_8bit=args.adam_8bit)

    # Live progress: log each eval + a heartbeat AS THEY HAPPEN (stdout + a JSONL on the persistent
    # checkpoint disk + optional Weights & Biases) -- so the run is monitorable mid-flight, not only
    # at the end. metrics.jsonl is tail-able; wandb gives a browser/phone dashboard.
    run_meta = {"features": args.features, "span": span_eff, "k": args.k, "batch": args.batch,
                "steps": args.steps, "params_b": round(total / 1e9, 2), "adam_8bit": args.adam_8bit}
    wb = None
    if args.wandb:
        import wandb  # noqa: PLC0415
        wb = wandb.init(project=args.wandb_project, config=run_meta)
    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    metrics_f = open(Path(args.ckpt) / "metrics.jsonl", "a")  # noqa: SIM115

    def progress(rec: dict) -> None:
        if "fvu_image" in rec:  # full eval
            log(f"  step {rec['step']}  recon_mse {rec.get('recon_mse', 0):.4f}  "
                f"fvu_text {rec.get('fvu_text', float('nan')):.3f}  "
                f"fvu_image {rec.get('fvu_image', float('nan')):.3f}  l0 {rec.get('l0', 0):.1f}")
        else:  # heartbeat
            log(f"  [hb] step {rec['step']}  recon_mse {rec.get('recon_mse', 0):.4f}")
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        if wb is not None:
            wb.log(rec, step=rec["step"])

    log(f"training {args.steps} steps, batch {args.batch}, streaming buffer {args.buffer_shards} "
        f"shards; checkpointing to {args.ckpt}/clt_ckpt.pt every {args.checkpoint_every}; "
        f"metrics -> {args.ckpt}/metrics.jsonl{' + wandb' if wb else ''}...")
    out = train_clt_streaming(
        args.cache, cfg, steps=args.steps, batch_size=args.batch, eval_every=args.eval_every,
        test_frac=args.test_frac, max_test_tokens=args.max_test_tokens,
        buffer_shards=args.buffer_shards, checkpoint_dir=args.ckpt,
        checkpoint_every=args.checkpoint_every,
        progress_fn=progress, heartbeat_every=args.heartbeat_every,
    )
    metrics_f.close()
    if wb is not None:
        wb.finish()
    log("FINAL scorecard:", {k: round(v, 4) for k, v in out["scorecard"].items()})
    Path(args.out).write_text(json.dumps({
        "layers": layers, "d_model": d_model, "n_features": args.features, "span": span_eff,
        "k": args.k, "adam_8bit": args.adam_8bit, "params": total, "steps": args.steps,
        "tokens": tokens, "scorecard": out["scorecard"], "history": out["history"],
    }, indent=2))
    log(f"saved {args.out}; CLT checkpoint at {args.ckpt}/clt_ckpt.pt")
    log("=== DONE clt_scale ===")


if __name__ == "__main__":
    main()
