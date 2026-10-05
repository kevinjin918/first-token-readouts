"""Stage 1 / A1 (reframed) — train a small transcoder on MedGemma's OWN activations.

The stock Gemma Scope transcoders won't apply cleanly to MedGemma (circuit-tracer's VLM wall
+ convention coupling). So we answer A1's real question in a convention we control AND
prototype the CLT training pipeline:

  1. Capture MedGemma MLP (in, out) activations across the depth.
  2. Train a small TopK transcoder on TEXT-token activations (mlp_in -> mlp_out).
  3. Evaluate reconstruction FVU on held-out TEXT tokens vs IMAGE tokens.

If text FVU is low (transcoder fits text) and image FVU is high, image-token activations are
out-of-distribution for a text-trained dictionary -> the measured justification for an
image-token-aware CLT, on MedGemma specifically. A ridge least-squares linear map is included
as a robust reference. Convention: mlp_in = layer.mlp input (post pre_feedforward_layernorm),
mlp_out = layer.mlp output (raw) — clean, well-defined HF tensors.

Hardened run: every 3rd layer (0..33), averaged over many CXRs/reports, results saved to JSON.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "google/medgemma-1.5-4b-it"
CXR_GLOB = str(Path.home() / ".cache/tracecxr/data/chestxray14/images/*.png")
CHEXPERT_CSV = str(Path.home() / ".cache/tracecxr/data/chexpert_plus/df_chexpert_plus_240401.csv")
OUT_JSON = str(Path.home() / "a1_fidelity.json")
LAYERS = list(range(0, 34, 3))  # 0,3,...,33 — the depth curve
D_FEAT, K, STEPS, BATCH, LR = 8192, 32, 4000, 4096, 1e-3
N_TEXT, N_IMG, MAXTOK = 400, 120, 96
IMG_ID = 262144


def log(*a: object) -> None:
    print(*a, flush=True)


class TopKTranscoder(nn.Module):
    def __init__(self, d: int, f: int, k: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d, f) * (d ** -0.5))
        self.b_enc = nn.Parameter(torch.zeros(f))
        self.W_dec = nn.Parameter(torch.randn(f, d) * (f ** -0.5))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        val, idx = pre.topk(self.k, dim=-1)
        feats = torch.zeros_like(pre).scatter_(-1, idx, F.relu(val))
        return feats @ self.W_dec + self.b_dec


def fvu(true: torch.Tensor, recon: torch.Tensor) -> float:
    num = ((true - recon) ** 2).sum()
    den = ((true - true.mean(0, keepdim=True)) ** 2).sum()
    return (num / den).item()


def ridge_map(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Least-squares y ~= x @ W via ridge normal equations (robust; GPU lstsq returns NaN)."""
    d = x.shape[1]
    xtx = x.T @ x
    reg = lam * xtx.diag().mean() * torch.eye(d, device=x.device, dtype=x.dtype)
    return torch.linalg.solve(xtx + reg, x.T @ y)


def report_texts(n: int) -> list[str]:
    import pandas as pd

    df = pd.read_csv(CHEXPERT_CSV)
    col = "report" if "report" in df.columns else df.columns[0]
    texts: list[str] = []
    for v in df[col].tolist():
        if isinstance(v, str) and len(v) > 40:
            texts.append(v[:1500])
        if len(texts) >= n:
            break
    return texts


def main() -> None:
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log("loading MedGemma...")
    proc = AutoProcessor.from_pretrained(MODEL)
    m = (
        AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
        .to("cuda").eval()
    )
    layers = m.model.language_model.layers

    cap: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def mk(i: int):
        def hook(_mod, inp, o):  # noqa: ANN001
            oo = o[0] if isinstance(o, tuple) else o
            cap[i] = (inp[0].detach()[0], oo.detach()[0])  # (seq, d), bf16, gpu
        return hook

    handles = [layers[i].mlp.register_forward_hook(mk(i)) for i in LAYERS]
    buf = {L: {"ti": [], "to": [], "ii": [], "io": []} for L in LAYERS}

    def forward_capture(messages):
        cap.clear()
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            m(**inputs)
        return inputs["input_ids"][0]

    log(f"gathering TEXT activations ({N_TEXT} reports)...")
    for txt in report_texts(N_TEXT):
        ids = forward_capture([{"role": "user", "content": [{"type": "text", "text": txt}]}])
        sel = slice(0, min(ids.numel(), MAXTOK))
        for L in LAYERS:
            mi, mo = cap[L]
            buf[L]["ti"].append(mi[sel].cpu())  # keep bf16 on cpu
            buf[L]["to"].append(mo[sel].cpu())

    log(f"gathering IMAGE activations ({N_IMG} CXRs)...")
    for p in sorted(glob.glob(CXR_GLOB))[:N_IMG]:
        pil = Image.open(p).convert("RGB")
        ids = forward_capture([{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": "Interpret this chest X-ray."}]}])
        img_mask = (ids == IMG_ID).cpu()
        for L in LAYERS:
            mi, mo = cap[L]
            buf[L]["ii"].append(mi.cpu()[img_mask])
            buf[L]["io"].append(mo.cpu()[img_mask])
    for h in handles:
        h.remove()

    # free the model -> full GPU for training
    del m
    torch.cuda.empty_cache()

    log("\nlayer | trainTOK testTXT imgTOK | linTXT linIMG linRATIO | tcTXT  tcIMG  tcRATIO")
    results = []
    for L in LAYERS:
        ti = torch.cat(buf[L]["ti"]).cuda().float()
        to = torch.cat(buf[L]["to"]).cuda().float()
        ii = torch.cat(buf[L]["ii"]).cuda().float()
        io = torch.cat(buf[L]["io"]).cuda().float()
        n = ti.shape[0]
        ntr = int(n * 0.85)
        perm = torch.randperm(n, device="cuda")
        tr, te = perm[:ntr], perm[ntr:]

        W = ridge_map(ti[tr], to[tr])
        lin_txt, lin_img = fvu(to[te], ti[te] @ W), fvu(io, ii @ W)

        tc = TopKTranscoder(ti.shape[1], D_FEAT, K).cuda()
        opt = torch.optim.Adam(tc.parameters(), lr=LR)
        for _ in range(STEPS):
            b = tr[torch.randint(0, ntr, (BATCH,), device="cuda")]
            opt.zero_grad()
            F.mse_loss(tc(ti[b]), to[b]).backward()
            opt.step()
        with torch.inference_mode():
            tc_txt, tc_img = fvu(to[te], tc(ti[te])), fvu(io, tc(ii))
        log(f"  {L:3d} | {ntr:7d} {len(te):6d} {io.shape[0]:6d} |"
            f" {lin_txt:6.3f} {lin_img:6.3f} {lin_img/lin_txt:7.2f} |"
            f" {tc_txt:6.3f} {tc_img:6.3f} {tc_img/tc_txt:7.2f}")
        results.append({"layer": L, "lin_text_fvu": lin_txt, "lin_image_fvu": lin_img,
                        "tc_text_fvu": tc_txt, "tc_image_fvu": tc_img,
                        "n_image_tokens": int(io.shape[0])})
        del ti, to, ii, io, tc, opt
        torch.cuda.empty_cache()

    mt = sum(r["tc_text_fvu"] for r in results) / len(results)
    mi_ = sum(r["tc_image_fvu"] for r in results) / len(results)
    summary = {"mean_text_fvu": mt, "mean_image_fvu": mi_, "image_text_ratio": mi_ / mt,
               "n_text_reports": N_TEXT, "n_images": N_IMG, "d_features": D_FEAT, "k": K,
               "steps": STEPS, "layers": LAYERS, "per_layer": results}
    Path(OUT_JSON).write_text(json.dumps(summary, indent=2))
    log(f"\n=== summary === mean text FVU {mt:.3f} | image FVU {mi_:.3f} | ratio {mi_/mt:.2f}")
    log(f"saved {OUT_JSON}")
    log("=== DONE train_transcoder ===")


if __name__ == "__main__":
    main()
