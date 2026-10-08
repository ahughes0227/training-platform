from __future__ import annotations

import io
import json
import sys
import types

import pytest
from PIL import Image

from defect_platform.contracts import (
    CleanlabSpec,
    DatasetSpec,
    LabelSource,
    ObjectSpec,
    SourceKind,
    SplitSpec,
)
from defect_platform.dataset import LabelRow, build_dataset, verify_dataset_version


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(output, format="PNG")
    return output.getvalue()


def _install_fake_cleanlab(monkeypatch):
    # The fake stands in for cleanlab, but the analysis itself needs numpy.
    pytest.importorskip("numpy")

    def find_label_issues(labels, pred_probs, *, return_indices_ranked_by, n_jobs):
        assert return_indices_ranked_by == "self_confidence"
        assert n_jobs == 1
        return [1]

    package = types.ModuleType("cleanlab")
    filter_module = types.ModuleType("cleanlab.filter")
    filter_module.find_label_issues = find_label_issues
    package.filter = filter_module
    monkeypatch.setitem(sys.modules, "cleanlab", package)
    monkeypatch.setitem(sys.modules, "cleanlab.filter", filter_module)


def _fixture(tmp_path, *, threshold: float, key_field: str = "image_uri"):
    object_spec = ObjectSpec(slug="panel", display_name="Panel", classes=["crack", "dent"])
    predictions = []
    rows = []
    for index, label in enumerate(["crack", "crack", "dent", "dent"]):
        image = tmp_path / f"image-{index}.png"
        image.write_bytes(_png((index * 30, 20, 100)))
        sample_id = f"sample-{index}"
        rows.append(LabelRow(str(image), label, sample_id=sample_id, source="fixture", row_number=index + 2))
        probabilities = [0.9, 0.1] if label == "crack" else [0.1, 0.9]
        if index == 1:
            probabilities = [0.1, 0.9]
        predictions.append({"key": sample_id if key_field == "sample_id" else str(image), "probabilities": probabilities})
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps({
        "format_version": 1,
        "object_slug": "panel",
        "classes": ["crack", "dent"],
        "key_field": key_field,
        "out_of_sample": True,
        "predictions": predictions,
    }), encoding="utf-8")
    spec = DatasetSpec(
        object_slug="panel",
        sources=[LabelSource(kind=SourceKind.CSV, location="unused.csv")],
        split=SplitSpec(train=0.5, validation=0.25, test=0.25, seed=7),
        output_uri=str(tmp_path / "datasets"),
        cleanlab=CleanlabSpec(predictions_uri=str(predictions_path), key_field=key_field,
                              max_issue_fraction=threshold),
    )
    return spec, object_spec, rows


def test_cleanlab_report_is_published_and_verified(tmp_path, monkeypatch):
    _install_fake_cleanlab(monkeypatch)
    spec, object_spec, rows = _fixture(tmp_path, threshold=0.25, key_field="sample_id")

    version = build_dataset(spec, object_spec, rows=rows, near_duplicate_distance=0)
    report_path = tmp_path / "datasets" / version.version_id / "cleanlab" / "report.json"
    report = json.loads(report_path.read_text())

    assert report["issue_count"] == 1
    assert report["issue_fraction"] == 0.25
    verify_dataset_version(version, object_spec.classes)


def test_cleanlab_gate_fails_before_immutable_publication(tmp_path, monkeypatch):
    _install_fake_cleanlab(monkeypatch)
    spec, object_spec, rows = _fixture(tmp_path, threshold=0.0)

    with pytest.raises(ValueError, match="Cleanlab quality gate failed"):
        build_dataset(spec, object_spec, rows=rows, near_duplicate_distance=0)
    assert not list((tmp_path / "datasets").glob("*/_COMMIT.json"))


def test_cleanlab_rejects_prediction_key_mismatch(tmp_path, monkeypatch):
    _install_fake_cleanlab(monkeypatch)
    spec, object_spec, rows = _fixture(tmp_path, threshold=0.25)
    predictions_path = tmp_path / "predictions.json"
    payload = json.loads(predictions_path.read_text())
    payload["predictions"][0]["key"] = "missing"
    predictions_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="prediction keys do not match"):
        build_dataset(spec, object_spec, rows=rows, near_duplicate_distance=0)
