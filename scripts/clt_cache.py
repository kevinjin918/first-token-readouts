"""Parallel activation caching for the max single-H100 CLT (docs/clt_spec.md).

Running MedGemma over the corpus is the slow, embarrassingly-parallel stage. This script
splits the (deterministically shuffled) corpus across ``--workers`` 1-GPU VMs: each caches its
contiguous slice into ``<out>/worker_<W>/`` (spot-safe via ``cache_corpus(resume=True)``), and a
final ``--merge`` pass writes one top-level ``<out>/manifest.json`` referencing the per-worker
shards as ``worker_<W>/shard_*.npz``. The loaders resolve ``cache_dir / name`` directly, so the
merged cache is read exactly like a single-worker one — no file moving.

The corpus is built identically in every invocation (same seed, same ``--n-text/--n-img``), so
each worker agrees on the global order before taking its slice. **Pass the same --n-text/--n-img
to every worker and to --merge.**

Records hold the image *path* (not a loaded PIL), so slicing is cheap and resume-skipped records
never touch disk; the capture wrapper opens the image only when a record is actually processed.

Disk: all 34 layers in bf16 cost ~0.33 MB/token (tokens x layers x 2 streams x d_model x 2 B);
~3 M tokens ~ 1 TB. Size --n-text/--n-img to the cache disk (see docs/clt_spec.md).

Usage:
  # on each of the 3 VMs (W = 0,1,2):
  python scripts/clt_cache.py --worker W --workers 3 --n-text 8000 --n-img 8000
  # then, on any one VM once all workers finish:
  python scripts/clt_cache.py --merge --n-text 8000 --n-img 8000
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from types import SimpleNamespace

from tracecxr.transcoder.cache import cache_corpus, load_manifest, medgemma_capture_fn

CACHE = Path.home() / "clt_scale_cache"
CXR_GLOB = str(Path.home() / ".cache/tracecxr/data/chestxray14/images/*.png")
CSV = str(Path.home() / ".cache/tracecxr/data/chexpert_plus/df_chexpert_plus_240401.csv")
SEED = 0


def log(*a: object) -> None:
    print(*a, flush=True)


def parse_layers(spec: str) -> list[int]:
    """``"0-33"`` -> ``[0, 1, ..., 33]``; ``"8-19"`` -> the B1 mid-block."""
    lo, hi = (int(x) for x in spec.split("-"))
    return list(range(lo, hi + 1))


def build_corpus(n_text: int, n_img: int) -> list[SimpleNamespace]:
    """The full, deterministically-shuffled corpus (lightweight records; images by path)."""
    import pandas as pd  # noqa: PLC0415

    reports = [r for r in pd.read_csv(CSV)["report"].tolist() if isinstance(r, str) and len(r) > 40]
    text_recs = [
        SimpleNamespace(id=f"t{i}", text=r[:1500], image_path=None)
        for i, r in enumerate(reports[:n_text])
    ]
    img_paths = sorted(glob.glob(CXR_GLOB))[:n_img]
    img_recs = [
        SimpleNamespace(id=f"i{i}", text="Interpret this chest X-ray.", image_path=p)
        for i, p in enumerate(img_paths)
    ]
    records = text_recs + img_recs
    random.Random(SEED).shuffle(records)  # mix kinds across shards (streaming loader assumes it)
    return records


def loading_capture(layers: list[int]):
    """Wrap the MedGemma capture so records carry image *paths*, opened lazily per record."""
    base = medgemma_capture_fn(layers, store_bf16=True)  # half the disk, lossless vs the bf16 CLT

    def cap(rec: SimpleNamespace):
        img = None
        if rec.image_path is not None:
            from PIL import Image  # noqa: PLC0415

            img = Image.open(rec.image_path).convert("RGB")
        return base(SimpleNamespace(id=rec.id, text=rec.text, image=img))

    return cap


def run_worker(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    records = build_corpus(args.n_text, args.n_img)
    n = len(records)
    per = -(-n // args.workers)  # ceil division -> contiguous, near-equal slices
    lo, hi = args.worker * per, min((args.worker + 1) * per, n)
    my = records[lo:hi]
    out = Path(args.out) / f"worker_{args.worker}"
    log(f"worker {args.worker}/{args.workers}: records[{lo}:{hi}] = {len(my)} of {n}; "
        f"layers {layers[0]}-{layers[-1]} ({len(layers)}); out {out}")
    if not my:
        log("nothing to cache for this worker (workers > records?)")
        return
    cap = loading_capture(layers)
    man = cache_corpus(my, layers, out, cap, shard_size=args.shard_size, resume=True)
    log(f"worker {args.worker} done: {len(man['shards'])} shards, tokens {man['token_counts']}")


def run_merge(args: argparse.Namespace) -> None:
    out = Path(args.out)
    worker_dirs = sorted(d for d in out.glob("worker_*") if (d / "manifest.json").exists())
    if not worker_dirs:
        raise SystemExit(f"no worker_*/manifest.json under {out} — run the workers first")
    merged_shards: list[str] = []
    token_counts: dict[str, int] = {}
    n_records = 0
    layers: list[int] | None = None
    shard_size: int | None = None
    for wd in worker_dirs:
        m = load_manifest(wd)
        if layers is None:
            layers, shard_size = m["layers"], m["shard_size"]
        elif m["layers"] != layers:
            raise SystemExit(f"layer mismatch: {wd.name} has {m['layers']}, expected {layers}")
        merged_shards.extend(f"{wd.name}/{name}" for name in m["shards"])
        for k, v in m["token_counts"].items():
            token_counts[k] = token_counts.get(k, 0) + int(v)
        n_records += int(m["n_records"])
    manifest = {
        "layers": layers, "shards": merged_shards, "n_records": n_records,
        "shard_size": shard_size, "token_counts": token_counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"merged {len(worker_dirs)} workers -> {len(merged_shards)} shards, "
        f"{n_records} records, tokens {token_counts}")
    log(f"wrote {out / 'manifest.json'} — train with scripts/clt_scale.py")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=int, help="this worker's index (0-based)")
    p.add_argument("--workers", type=int, default=3, help="total parallel workers")
    p.add_argument("--merge", action="store_true", help="merge per-worker caches into one manifest")
    p.add_argument("--n-text", type=int, default=8000, help="number of report (text) records")
    p.add_argument("--n-img", type=int, default=8000, help="number of CXR (image) records")
    p.add_argument("--layers", default="0-33", help="inclusive layer range, e.g. 0-33 (all 34)")
    p.add_argument("--shard-size", type=int, default=32, help="records per shard")
    p.add_argument("--out", default=str(CACHE), help="cache root directory")
    args = p.parse_args()

    if args.merge:
        run_merge(args)
    elif args.worker is not None:
        if not 0 <= args.worker < args.workers:
            raise SystemExit(f"--worker must be in [0, {args.workers})")
        run_worker(args)
    else:
        raise SystemExit("pass --worker W (to cache a slice) or --merge (to combine)")


if __name__ == "__main__":
    main()
