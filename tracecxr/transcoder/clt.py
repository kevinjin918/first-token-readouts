"""Stage 1 cross-layer transcoder (CLT) — implementation (manifesto §Stage 1).

The interpretable replacement model: a bank of sparse features that mimics MedGemma's decoder
MLPs. The **cross-layer** property — a feature at layer ``L`` writes into the reconstruction of
later layers — collapses multi-layer computation into single nameable features and shortens
attribution-graph paths (the paper reports average path length 3.7 -> 2.3), which is what
makes the downstream graphs (Stage 2) legible.

Memory makes the full cross-layer decoder intractable to materialize densely at scale (each
feature would need a decoder vector for every downstream layer). So this implements a
**configurable-span** CLT: a feature at layer ``s`` writes to layers ``s … s+span``.
``span=0`` is a per-layer transcoder (validated in ``results/a3_imageaware.md``); a small
``span`` is the windowed cross-layer that fits a single L4 / a few A100s; ``span=None`` is the
full CLT (feasible only at small sizes, or via sparse kernels — ``EleutherAI/clt-training``).

Encoder nonlinearity is **TopK** (BatchTopK-style) — the proven path from A3; it makes L0
exactly ``k`` and needs no threshold tuning. (``CLTConfig.jumprelu`` is kept for compat; the
working encoder is TopK.) Warm-start from the released Gemma Scope transcoders is *not* used —
A1 showed their activation convention is incompatible with our raw-HF capture, and A3 showed
training from scratch reaches text-level fidelity — so :meth:`load_warm_start` stays
unimplemented and training is from scratch.

torch is imported lazily (on first build), so importing this package stays offline.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from tracecxr.core import config

_NOT_IMPLEMENTED_MSG = "Stage 1 not yet implemented; see TraceCXR_manifesto.md §Stage 1"

#: Default warm-start locator: the released Gemma3-4B-IT transcoder weights (UNVERIFIED, unused).
_DEFAULT_WARM_START = config.TRANSCODERS["gemma3-4b-it-clt"].name


@dataclass
class CLTConfig:
    """Hyperparameters for the Stage 1 cross-layer transcoder.

    Attributes:
        jumprelu: Intended JumpReLU encoder (manifesto). Kept for compat; the working encoder
            is TopK (see module docstring), so this flag is currently informational.
        lambda_: Sparsity penalty coefficient (a JumpReLU-path knob; TopK sparsity is structural).
        c: Tanh scale inside the sparsity penalty.
        learning_rate: Adam learning rate.
        batch_size: Training batch size (token-activations per step).
        n_features: Width of the sparse feature dictionary per layer.
        n_layers: Number of decoder MLP layers the CLT replaces / writes into.
        l0_target: Target active features per token (sanity target). With TopK, L0 == ``k``.
        recon_error_target: Target normalized reconstruction error (manifesto: under ~0.2).
        warm_start_from: Released-transcoder locator (NOT used — see module docstring).
        span: Cross-layer write span — a feature at layer ``s`` writes to ``s … s+span``.
            ``0`` = per-layer; small int = windowed; ``None`` = full cross-layer.
        k: TopK active features per token (the structural sparsity).
        device: Torch device override (None -> cuda if available else cpu).
        amp: Use bf16 mixed precision for the matmuls on CUDA (H100 tensor cores; halves
            activation memory). The loss is still computed in fp32 for stability. No-op on CPU.
        adam_8bit: Use the bitsandbytes 8-bit Adam optimizer instead of ``torch.optim.Adam``.
            Halves optimizer-state memory (fp32 m+v = 8 B/param -> 2 B/param), which lets a much
            larger CLT fit one 80 GB H100 (see ``docs/clt_spec.md``). Falls back to torch Adam with
            a warning if bitsandbytes isn't installed. Resume only with the *same* setting (the
            saved optimizer state is type-specific).
        extra: Escape hatch for additional knobs read at use-time.
    """

    jumprelu: bool = True
    lambda_: float = 1e-3
    c: float = 4.0
    learning_rate: float = 1e-4
    batch_size: int = 2048
    n_features: int = 131_072
    n_layers: int = 34
    l0_target: int = 100
    recon_error_target: float = 0.2
    warm_start_from: str | None = _DEFAULT_WARM_START
    span: int | None = 1
    k: int = 32
    device: str | None = None
    amp: bool = False
    adam_8bit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive, got {self.n_features}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.l0_target <= 0:
            raise ValueError(f"l0_target must be positive, got {self.l0_target}")
        if not 0.0 <= self.recon_error_target <= 1.0:
            raise ValueError(
                f"recon_error_target must be in [0, 1], got {self.recon_error_target}"
            )
        if self.span is not None and self.span < 0:
            raise ValueError(f"span must be None or >= 0, got {self.span}")
        if not 0 < self.k <= self.n_features:
            raise ValueError(f"k must be in (0, n_features], got {self.k}")


class CrossLayerTranscoder:
    """Sparse, configurable-span cross-layer replacement for MedGemma's decoder MLPs.

    Holds a :class:`CLTConfig` and lazily builds its parameters on first use (so import is
    offline). :meth:`encode` produces per-(layer, feature) activations; :meth:`decode`
    reconstructs every layer's MLP output with each feature writing to its own layer and the
    next ``span`` layers; :meth:`reconstruct` chains them; :meth:`train_step` runs one Adam
    step minimizing per-layer reconstruction MSE (TopK supplies the sparsity).
    """

    def __init__(self, clt_config: CLTConfig | None = None) -> None:
        self.config: CLTConfig = clt_config if clt_config is not None else CLTConfig()
        self._params: dict[str, Any] | None = None
        self._opt: Any = None
        self._d_model: int | None = None
        self._span: int | None = None

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"{type(self).__name__}(n_features={cfg.n_features}, n_layers={cfg.n_layers}, "
            f"span={cfg.span}, k={cfg.k}, jumprelu={cfg.jumprelu}, "
            f"warm_start_from={cfg.warm_start_from!r})"
        )

    # -- lazy parameter build -----------------------------------------------------------

    def _device(self) -> str:
        import torch  # noqa: PLC0415

        if self.config.device is not None:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_built(self, d_model: int, with_optimizer: bool = True) -> None:
        if self._params is not None:
            return
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        cfg = self.config
        n_layers, n_features = cfg.n_layers, cfg.n_features
        span = (n_layers - 1) if cfg.span is None else min(int(cfg.span), n_layers - 1)
        dev = self._device()
        self._d_model, self._span = d_model, span
        enc = torch.randn(n_layers, d_model, n_features, device=dev) * d_model**-0.5
        self._params = {
            "W_enc": nn.Parameter(enc),
            "b_enc": nn.Parameter(torch.zeros(n_layers, n_features, device=dev)),
            "b_dec": nn.Parameter(torch.zeros(n_layers, d_model, device=dev)),
        }
        # Decoder: ONE tensor per write-offset, each (source layer, feature, d_model) — offset o
        # holds the weights writing a source layer's features into its layer+o. A single
        # (n_layers, span+1, n_features, d_model) tensor would, at high span/features, exceed 2^31
        # elements and break bitsandbytes' 8-bit-Adam kernel (int32 indexing). Per-offset tensors
        # are n_layers*n_features*d_model each (independent of span), under the limit for any
        # practical config — so span (the cross-layer reach) is no longer bounded by the optimizer.
        for offset in range(span + 1):
            dec = torch.randn(n_layers, n_features, d_model, device=dev) * n_features**-0.5
            self._params[f"W_dec_{offset}"] = nn.Parameter(dec)
        # The 8-bit-Adam state is ~2 B/param (~12 GB for the 6.24 B CLT); inference (eval, ablation)
        # never steps the optimizer, so skip building it — saves the memory and the references that
        # would otherwise block freeing between sequentially-loaded models.
        if with_optimizer:
            self._opt = self._build_optimizer(list(self._params.values()))

    def _build_optimizer(self, params: list[Any]) -> Any:
        """Adam optimizer — bitsandbytes 8-bit when ``adam_8bit`` (halves optimizer memory)."""
        import torch  # noqa: PLC0415

        lr = self.config.learning_rate
        if self.config.adam_8bit:
            try:
                import bitsandbytes as bnb  # noqa: PLC0415

                return bnb.optim.Adam8bit(params, lr=lr)
            except ImportError:
                import warnings  # noqa: PLC0415

                warnings.warn(
                    "adam_8bit=True but bitsandbytes is not installed; falling back to "
                    "torch.optim.Adam (more optimizer memory). pip install bitsandbytes",
                    stacklevel=2,
                )
        return torch.optim.Adam(params, lr=lr)

    def _as_tensor(self, x: Any) -> Any:
        import torch  # noqa: PLC0415

        # Preserve an existing tensor's dtype (so bf16-preloaded activations stay bf16);
        # only numpy/other inputs are coerced to float32.
        if isinstance(x, torch.Tensor):
            return x.to(self._device())
        return torch.as_tensor(x, dtype=torch.float32, device=self._device())

    def _autocast(self):
        """bf16 autocast context on CUDA when ``amp`` is set; a no-op otherwise."""
        import torch  # noqa: PLC0415

        if self.config.amp and self._device() == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # -- core compute -------------------------------------------------------------------

    def encode(self, residuals: Any) -> Any:
        """Per-layer residual activations -> sparse TopK feature activations.

        Args:
            residuals: ``(..., n_layers, d_model)`` activations (numpy or torch).

        Returns:
            ``(..., n_layers, n_features)`` activations (exactly ``k`` active per layer).
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        x = self._as_tensor(residuals)
        self._ensure_built(x.shape[-1])
        pre = torch.einsum("...ld,ldf->...lf", x, self._params["W_enc"]) + self._params["b_enc"]
        val, idx = pre.topk(self.config.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, F.relu(val))

    def decode(self, feature_acts: Any) -> Any:
        """Sparse feature activations -> reconstructed per-layer MLP outputs (cross-layer).

        A feature at source layer ``s`` contributes to the reconstruction of layers
        ``s … s+span`` via the corresponding decoder offset.

        Args:
            feature_acts: ``(..., n_layers, n_features)``.

        Returns:
            ``(..., n_layers, d_model)`` reconstructed MLP outputs.
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        feats = self._as_tensor(feature_acts)
        if self._params is None:
            raise RuntimeError("decode called before build; call encode/reconstruct first")
        n_layers = self._params["W_dec_0"].shape[0]
        span = self._span or 0
        # Accumulate per-offset contributions with a running sum instead of stacking them: a stack
        # of span+1 copies of (..., n_layers, d) peaks at (span+1)x that memory — the dominant cost
        # at high span — whereas a running sum keeps only ~2 live at once.
        out = None
        for offset in range(span + 1):
            src = feats[..., : n_layers - offset, :]  # sources 0..n_layers-1-offset
            dec = self._params[f"W_dec_{offset}"][: n_layers - offset]  # (n_src, F, d)
            c = torch.einsum("...lf,lfd->...ld", src, dec)  # targets offset..n_layers-1
            c = F.pad(c, (0, 0, offset, 0))  # pad layer axis front -> (..., L, d)
            out = c if out is None else out + c
        return self._params["b_dec"] + out

    def reconstruct(self, residuals: Any) -> Any:
        """Full encode-then-decode pass: residual stream -> reconstructed MLP outputs."""
        with self._autocast():
            return self.decode(self.encode(residuals))

    def train_step(self, batch: Any) -> dict[str, float]:
        """One Adam step on a ``(inputs, targets)`` batch of per-layer activations.

        Args:
            batch: ``(residuals, mlp_outputs)``, each ``(..., n_layers, d_model)`` — the MLP
                inputs to encode and the MLP outputs to reconstruct.

        Returns:
            ``{"loss", "recon_mse", "l0"}``.
        """
        import torch  # noqa: PLC0415

        x, y = batch
        x, y = self._as_tensor(x), self._as_tensor(y)
        self._ensure_built(x.shape[-1])
        self._opt.zero_grad()
        with self._autocast():
            feats = self.encode(x)
            recon = self.decode(feats)
        # Loss in fp32 for stable optimization even when the matmuls ran in bf16.
        recon_mse = ((recon.float() - y.float()) ** 2).mean()
        loss_val = float(recon_mse.detach())
        recon_mse.backward()
        self._opt.step()
        with torch.inference_mode():
            l0 = float((feats > 0).float().sum(-1).mean())
        return {"loss": loss_val, "recon_mse": loss_val, "l0": l0}

    def load_warm_start(self, source: str | None = None) -> None:
        """Not used — the released Gemma Scope convention is incompatible (A1), and training
        from scratch reaches text-level fidelity (A3). Kept on the interface for future support.
        """
        raise NotImplementedError(
            "Stage 1: warm-start from released transcoders is unsupported (A1 convention "
            "mismatch); train from scratch — see results/a1_fidelity.md and docs/stage1_plan.md"
        )

    # -- checkpointing (for preemptible / spot training) --------------------------------

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save params + optimizer + step so a preempted run can resume.

        Written atomically (tmp file then rename) so a preemption mid-write cannot corrupt the
        checkpoint. Requires the CLT to have been built (trained/encoded at least once).
        """
        from pathlib import Path  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if self._params is None:
            raise RuntimeError("nothing to checkpoint; build/train the CLT first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp"
        torch.save(
            {
                "step": int(step),
                "d_model": self._d_model,
                "params": {k: v.detach().cpu() for k, v in self._params.items()},
                "opt": self._opt.state_dict(),
            },
            tmp,
        )
        Path(tmp).replace(path)  # atomic on POSIX

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> int:
        """Restore params (+ optimizer) from :meth:`save_checkpoint`; returns the resume step.

        ``load_optimizer=False`` loads only the weights — for **inference** (eval / ablation /
        substitution), where the ~12 GB optimizer state is dead weight that can OOM the GPU. Resume
        training needs ``load_optimizer=True`` (the default).

        Only load checkpoints you wrote — this unpickles optimizer state (``weights_only=False``).
        A ``.safetensors`` path is a weights-only export (:mod:`tracecxr.transcoder.export`) and
        loads without pickle, one tensor at a time.
        """
        if str(path).endswith(".safetensors"):
            if load_optimizer:
                raise ValueError("a .safetensors export holds weights only; it cannot resume "
                                 "training (pass load_optimizer=False)")
            return self._load_safetensors(path)
        import torch  # noqa: PLC0415

        # Load to CPU, not the GPU: a large checkpoint (e.g. the 6.24 B CLT is ~36 GB) loaded with
        # map_location=cuda spikes GPU memory by the full file size *on top of* the freshly-built
        # params/optimizer — enough to OOM an 80 GB card (esp. with MedGemma also resident). Loading
        # to host RAM and copying each tensor over below keeps the GPU peak to params+optimizer.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._ensure_built(int(ckpt["d_model"]), with_optimizer=load_optimizer)
        for key, value in ckpt["params"].items():
            if key not in self._params:
                raise ValueError(
                    f"checkpoint param {key!r} not in this CLT — the decoder layout changed "
                    "(W_dec is now split into per-offset W_dec_<o> tensors); retrain from scratch"
                )
            cur = self._params[key]
            if tuple(cur.shape) != tuple(value.shape):
                raise ValueError(
                    f"checkpoint param {key!r} shape {tuple(value.shape)} != current "
                    f"{tuple(cur.shape)} — the config (n_features/span/n_layers) changed since "
                    "this checkpoint was written; resume with the original config or start fresh"
                )
            cur.data.copy_(value.to(self._device()))
        if load_optimizer:
            self._opt.load_state_dict(ckpt["opt"])
        return int(ckpt["step"])

    def _load_safetensors(self, path: str) -> int:
        """Load a weights-only export; decoder blocks may hold only the rows decode reads."""
        from safetensors import safe_open  # noqa: PLC0415

        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            self._ensure_built(int(meta["d_model"]), with_optimizer=False)
            for key in f.keys():
                if key not in self._params:
                    raise ValueError(f"export tensor {key!r} not in this CLT; check the config")
                cur, value = self._params[key], f.get_tensor(key)
                trimmed = (key.startswith("W_dec_") and value.shape[1:] == cur.shape[1:]
                           and value.shape[0] <= cur.shape[0])
                if tuple(cur.shape) != tuple(value.shape) and not trimmed:
                    raise ValueError(f"export tensor {key!r} shape {tuple(value.shape)} != "
                                     f"current {tuple(cur.shape)}; check n_features/span/n_layers")
                # rows past the export's are never read by decode; zero them, not leave random
                cur.data.zero_()
                cur.data[: value.shape[0]].copy_(value.to(self._device()))
        return int(meta.get("step", 0))
