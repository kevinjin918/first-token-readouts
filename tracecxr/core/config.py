"""Central registry of external resource identifiers and local paths.

Resource ids for gated models and datasets live here behind a single module so that no
unit hardcodes a fragile URL inline. These values are *unverified* (several upstream
papers/repos are recent and move weekly) and are resolved only when real downloads are
explicitly requested — the mock-first test path never touches them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceId:
    """A pointer to an external artifact. ``verified`` flags whether we confirmed it."""

    name: str
    locator: str
    kind: str  # "hf-model" | "hf-dataset" | "physionet" | "url" | "github"
    gated: bool = False
    verified: bool = False
    note: str = ""


# --- Models (Unit 1). Confirm ids at use-time before downloading. ---
MODELS: dict[str, ResourceId] = {
    "medgemma": ResourceId(
        "medgemma", "google/medgemma-1.5-4b-it", "hf-model", gated=True, verified=True,
        note=(
            "Primary subject. MedGemma 1.5 4B-IT (Gemma-3 decoder); gated, requires accepting "
            "HAI-DEF terms. This is the variant CheXthought evaluated, so use it for "
            "comparability. The older google/medgemma-4b-it also exists — do not confuse them."
        ),
    ),
    "qwen3vl": ResourceId(
        "qwen3vl", "Qwen/Qwen3-VL-8B-Thinking", "hf-model", gated=False, verified=False,
        note="Secondary model; Apache-2.0. Verify exact id.",
    ),
    "chexagent": ResourceId(
        "chexagent", "StanfordAIMI/CheXagent-2-3b", "hf-model", gated=False, verified=False,
        note=(
            "Cross-model convergence probe (independently trained CXR VLM). CheXagent-2 3B "
            "(~2.78 B params, Phi-based) is the SIZE-MATCHED counterpart to MedGemma 1.5 4B — "
            "same ~3-4 B tier, so a convergence comparison isn't confounded by scale. The 8 B "
            "CheXagent would be ~2x and mix scale with architecture, so it is deliberately not "
            "used. Custom modeling code (trust_remote_code); from_list_format prompt API. "
            "Verify id + API at use-time."
        ),
    ),
}

# --- Datasets (Units 2-4). ---
DATASETS: dict[str, ResourceId] = {
    "chestxray14": ResourceId(
        "chestxray14", "https://nihcc.app.box.com/v/ChestXray-NIHCC", "url",
        note="Sparse bboxes in BBox_List_2017.csv (~984 imgs incl. Effusion, Pneumothorax).",
    ),
    "vindr": ResourceId(
        "vindr", "physionet.org/content/vindr-cxr", "physionet", gated=True,
        note="Dense radiologist boxes; PhysioNet credentialed + CITI.",
    ),
    "chexpert_plus": ResourceId(
        "chexpert_plus", "AIMI/chexpert_plus:5yyj", "redivis", gated=True, verified=True,
        note=(
            "Primary report-prior substrate (Redivis org AIMI, dataset chexpert_plus). "
            "Reports/paths in table df_chexpert_plus_240401 (223,462 rows); the 14 CheXpert "
            "labels live in separate label CSV files (the 'CheXpert Labels' file table), "
            "joined on path_to_image. Pull via scripts/download_chexpert_plus.py "
            "(REDIVIS_API_TOKEN). CheXthought traces align to these images."
        ),
    ),
    "chexthought": ResourceId(
        "chexthought", "StanfordAIMI/CheXthought", "hf-dataset", gated=True, verified=False,
        note="Cursor-trace coordinates for claim-region IoU (Unit 9). Verify release/terms.",
    ),
}

# --- Interpretability tooling (Units 15-19). ---
TRANSCODERS: dict[str, ResourceId] = {
    "gemma3-4b-it-clt": ResourceId(
        "gemma3-4b-it-clt", "UNVERIFIED", "hf-model", verified=False,
        note="Released Gemma3-4B-IT transcoder weights to warm-start from. Confirm repo id.",
    ),
}


def data_root() -> Path:
    """Local directory where real datasets are expected. Override with ``TRACECXR_DATA_DIR``."""
    return Path(os.environ.get("TRACECXR_DATA_DIR", Path.home() / ".cache" / "tracecxr" / "data"))


def cache_root() -> Path:
    """Local directory for model weights / intermediate artifacts."""
    return Path(os.environ.get("TRACECXR_CACHE_DIR", Path.home() / ".cache" / "tracecxr"))
