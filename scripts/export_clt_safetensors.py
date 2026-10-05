"""Export the released CLT checkpoint to the weights-only safetensors file on Hugging Face.

The Zenodo deposit (https://doi.org/10.5281/zenodo.22105454) holds the training checkpoint,
clt_ckpt.pt: 37,630,140,293 bytes, MD5 4c2747c75e4ce368bd48ccd194f9b8ef. It is a pickle carrying
the 6.24B allocated parameters plus the 8-bit Adam state. This writes the 3.30B parameters that
inference reads (W_enc, both biases and the decoder rows that land inside the model; see
tracecxr/transcoder/export.py), then reopens both files and checks every kept tensor is bitwise
equal to the checkpoint. CrossLayerTranscoder.load_checkpoint accepts either file.

Runs on CPU with a few GB of memory (the checkpoint is memory-mapped, tensors written one at a
time). Usage:
    python scripts/export_clt_safetensors.py --ckpt clt_ckpt.pt --out clt.safetensors
"""

from __future__ import annotations

import argparse
import json
import time

from tracecxr.transcoder.export import export_safetensors, verify_export


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="training checkpoint (clt_ckpt.pt)")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    args = ap.parse_args()
    if not args.out.endswith(".safetensors"):
        raise SystemExit("--out must end in .safetensors (load_checkpoint dispatches on it)")
    t0 = time.time()
    summary = export_safetensors(args.ckpt, args.out)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out} in {time.time() - t0:.0f} s; verifying", flush=True)
    n = verify_export(args.ckpt, args.out)
    print(f"verified: {n} tensors bitwise equal to {args.ckpt}")


if __name__ == "__main__":
    main()
