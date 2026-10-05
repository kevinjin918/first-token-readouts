"""Tests for materialize_image — lazy image loading from metadata['image_path']."""

from __future__ import annotations

import numpy as np
from PIL import Image
from tracecxr.core import CXRRecord, materialize_image


def test_returns_inmemory_image_unchanged():
    img = np.zeros((4, 4), dtype=np.uint8)
    rec = CXRRecord(id="a", image=img)
    assert materialize_image(rec) is img


def test_loads_from_path_when_image_none(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.full((5, 6), 200, dtype=np.uint8)).save(p)
    rec = CXRRecord(id="b", image=None, metadata={"image_path": str(p)})
    out = materialize_image(rec)
    assert out is not None and out.shape == (5, 6)


def test_none_when_no_image_and_no_path():
    assert materialize_image(CXRRecord(id="c", image=None)) is None


def test_none_when_path_missing(tmp_path):
    rec = CXRRecord(id="d", image=None, metadata={"image_path": str(tmp_path / "nope.png")})
    assert materialize_image(rec) is None
