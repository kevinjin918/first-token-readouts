"""Tests for the TraceCXR shared substrate (tracecxr.core)."""

from __future__ import annotations

import numpy as np
import pytest
from tracecxr.core import (
    BBox,
    CXRRecord,
    Finding,
    FixtureDataset,
    Label,
    Metric,
    MetricResult,
    MockVLM,
    ModelAdapter,
    ModelOutput,
    available_models,
    get_model,
    iter_metrics,
    load_dataset,
    register_metric,
)

# --- types ----------------------------------------------------------------------------

def test_bbox_iou_identical_is_one():
    b = BBox(0, 0, 10, 10)
    assert b.iou(b) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero():
    assert BBox(0, 0, 10, 10).iou(BBox(100, 100, 10, 10)) == 0.0


def test_bbox_iou_partial_overlap():
    # two 10x10 boxes overlapping in a 5x5 corner -> inter=25, union=175
    assert BBox(0, 0, 10, 10).iou(BBox(5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_bbox_rejects_negative_dims():
    with pytest.raises(ValueError):
        BBox(0, 0, -1, 5)


def test_cxrrecord_label_defaults_blank():
    rec = CXRRecord(id="x", image=None, findings={Finding.EFFUSION: Label.POSITIVE})
    assert rec.label(Finding.EFFUSION) == Label.POSITIVE
    assert rec.label(Finding.PNEUMOTHORAX) == Label.BLANK


def test_modeloutput_assertions():
    out = ModelOutput(
        text="", finding_predictions={Finding.EFFUSION: 0.9, Finding.PNEUMOTHORAX: 0.2}
    )
    assert out.asserts(Finding.EFFUSION)
    assert not out.asserts(Finding.PNEUMOTHORAX)
    assert out.asserted_findings() == {Finding.EFFUSION}


# --- mock model -----------------------------------------------------------------------

def test_mock_is_model_adapter():
    assert isinstance(MockVLM(), ModelAdapter)


def test_mock_registered():
    assert "mock" in available_models()
    assert isinstance(get_model("mock"), MockVLM)


def test_mock_report_prior_fires_without_image():
    """The default report prior (effusion) is asserted even with no image — the core probe."""
    model = MockVLM(report_prior=(Finding.EFFUSION,))
    rec = CXRRecord(id="r1", image=None, findings={Finding.EFFUSION: Label.NEGATIVE})
    out = model.generate(rec, with_image=False)
    assert out.asserts(Finding.EFFUSION)
    assert not out.abstained


def test_mock_honest_reads_image_labels():
    model = MockVLM(report_prior=(), honest=True)
    img = np.zeros((8, 8), dtype=np.uint8)
    rec = CXRRecord(id="r2", image=img, findings={Finding.PNEUMOTHORAX: Label.POSITIVE})
    assert model.generate(rec, with_image=True).asserts(Finding.PNEUMOTHORAX)
    # without the image and with no report prior, it abstains
    assert model.generate(rec, with_image=False).abstained


def test_mock_deterministic():
    model = MockVLM(report_prior=(), hallucination_rate=0.5)
    rec = CXRRecord(id="r3", image=None)
    assert model.generate(rec).finding_predictions == model.generate(rec).finding_predictions


# --- fixture dataset ------------------------------------------------------------------

def test_fixture_loads_and_has_focus_findings():
    ds = FixtureDataset()
    records = list(ds)
    assert len(records) == len(ds) >= 5
    # at least one positive effusion and one positive pneumothorax exist
    pos_eff = [r for r in records if r.label(Finding.EFFUSION) == Label.POSITIVE]
    pos_ptx = [r for r in records if r.label(Finding.PNEUMOTHORAX) == Label.POSITIVE]
    assert pos_eff and pos_ptx
    # positive findings carry a bbox and the image is loaded
    r = pos_eff[0]
    assert r.image is not None and r.image.ndim == 2
    assert r.bboxes[Finding.EFFUSION]


def test_load_dataset_fixture():
    ds = load_dataset("fixture")
    assert list(ds)


def test_load_dataset_unknown_raises():
    with pytest.raises(KeyError):
        load_dataset("does-not-exist")


# --- metric registry ------------------------------------------------------------------

class _DummyMetric:
    key = "dummy"

    def compute(self, model, dataset, **opts) -> MetricResult:
        n = sum(1 for _ in dataset)
        return MetricResult(key=self.key, value=float(n), n=n)


def test_metric_registry_roundtrip():
    m = register_metric(_DummyMetric())
    assert isinstance(m, Metric)
    assert m in iter_metrics(["dummy"])
    result = m.compute(MockVLM(), FixtureDataset())
    assert result.key == "dummy"
    assert result.n == len(FixtureDataset())
