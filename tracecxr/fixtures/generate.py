"""Generate the tiny synthetic CXR fixture used across the test suite.

Run with ``python -m tracecxr.fixtures.generate``. Produces ``manifest.json`` plus one
``.npy`` grayscale image per record, all committed to the repo so loading is offline and
deterministic. The images are not real X-rays — they are seeded noise with a bright
rectangle standing in for a "finding" region, sized to match the annotated bbox so the
occlusion and IoU units have something geometrically meaningful to operate on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracecxr.core.types import Finding, Label

OUT_DIR = Path(__file__).resolve().parent
IMG_SIZE = 64  # small, fast; real loaders resize to model input size

# Each record: id, image-level labels, and bbox (x, y, w, h) per positive focus finding.
_SPECS: list[dict] = [
    {"id": "fx_eff_pos", "report": "There is a moderate right pleural effusion.",
     "pos": {Finding.EFFUSION: (40, 44, 18, 16)}},
    {"id": "fx_ptx_pos", "report": "Large left apical pneumothorax.",
     "pos": {Finding.PNEUMOTHORAX: (6, 6, 16, 20)}},
    {"id": "fx_both_pos", "report": "Right effusion with associated pneumothorax.",
     "pos": {Finding.EFFUSION: (42, 46, 16, 14), Finding.PNEUMOTHORAX: (8, 8, 14, 18)}},
    {"id": "fx_normal_1", "report": "No acute cardiopulmonary process.", "pos": {}},
    {"id": "fx_normal_2", "report": "Clear lungs. No effusion or pneumothorax.", "pos": {}},
    {"id": "fx_eff_neg_ptx_pos", "report": "Apical pneumothorax, no effusion.",
     "pos": {Finding.PNEUMOTHORAX: (40, 6, 16, 18)}},
    {"id": "fx_eff_pos_2", "report": "Small left pleural effusion at the base.",
     "pos": {Finding.EFFUSION: (8, 46, 18, 14)}},
    {"id": "fx_normal_3", "report": "Unremarkable chest radiograph.", "pos": {}},
]

FOCUS = (Finding.EFFUSION, Finding.PNEUMOTHORAX)


def _make_image(seed: int, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((IMG_SIZE, IMG_SIZE)) * 80 + 40).astype(np.float32)
    for x, y, w, h in boxes:
        img[y : y + h, x : x + w] = 230.0  # bright "finding" region
    return img.astype(np.uint8)


def main() -> None:
    records = []
    for seed, spec in enumerate(_SPECS):
        pos: dict[Finding, tuple[int, int, int, int]] = spec["pos"]
        img = _make_image(seed, list(pos.values()))
        img_name = f"{spec['id']}.npy"
        np.save(OUT_DIR / img_name, img)

        findings = {}
        bboxes = {}
        for f in FOCUS:
            if f in pos:
                findings[f.value] = Label.POSITIVE.value
                x, y, w, h = pos[f]
                bboxes[f.value] = [{"x": x, "y": y, "w": w, "h": h}]
            else:
                findings[f.value] = Label.NEGATIVE.value

        records.append({
            "id": spec["id"],
            "image": img_name,
            "report": spec["report"],
            "findings": findings,
            "bboxes": bboxes,
            "metadata": {"synthetic": True},
        })

    manifest = {
        "name": "fixture",
        "description": "Synthetic CXR fixture for offline tests. Not real radiographs.",
        "image_size": IMG_SIZE,
        "focus_findings": [f.value for f in FOCUS],
        "records": records,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(records)} records + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
