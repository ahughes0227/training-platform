from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from types import SimpleNamespace

from PIL import Image
import pytest

from defect_platform.contracts import ModelRelease
from defect_platform.serve.api import PredictRequest, PredictionService
from defect_platform.serve.deploy import rayservice_manifest
from defect_platform.serve.releases import ReleaseManager
from defect_platform.semantics import legacy_catalog


def release(state: str = "staged") -> ModelRelease:
    return ModelRelease(
        object_slug="valves", model_name="defect-valves", model_version="2",
        release_id="release-2", run_id="run-2", dataset_version_id="data-1",
        runtime_id="runtime-1", serving_image_digest="repo/image@sha256:" + "a" * 64,
        state=state, approved_by="operator" if state == "promoted" else None,
        approved_at=datetime.now(timezone.utc) if state == "promoted" else None,
        catalog_sha256="b" * 64, model_semantic_sha256="c" * 64, bundle_sha256="d" * 64,
    )


def test_serving_decodes_image_and_returns_result():
    image = Image.new("RGB", (8, 8), "red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    catalog = legacy_catalog("valves", ["scratch", "dent"])
    service = PredictionService(
        lambda crop, explain: {"class_name": "scratch", "confidence": 0.92,
                               "review_required": False, "heatmap": crop if explain else None,
                               "class_id": catalog.class_ids[0], "catalog_sha256": catalog.sha256,
                               "semantic_sha256": "c" * 64},
        "defect-valves", "2",
        catalog=catalog, semantic_sha256="c" * 64,
    )
    response = service.predict(PredictRequest(image_base64=encoded, explain=True))
    assert response.class_name == "scratch"
    assert response.model_version == "2"
    assert response.heatmap_png_base64


def test_rayservice_requires_approval_and_exact_digest():
    with pytest.raises(ValueError):
        rayservice_manifest(release(), mlflow_tracking_uri="https://mlflow")
    result = rayservice_manifest(release("promoted"), mlflow_tracking_uri="https://mlflow",
                                 catalog_root="gs://catalogs")
    assert result["kind"] == "RayService"
    assert "@sha256:" in result["spec"]["rayClusterConfig"]["headGroupSpec"]["template"]["spec"]["containers"][0]["image"]


def test_promotion_requires_approver_and_moves_champion():
    class Registry:
        alias = None
        def verify_release(self, candidate):
            assert candidate.bundle_sha256 == "d" * 64
        def get_model_version(self, name, version):
            return SimpleNamespace(version=version)
        def set_registered_model_alias(self, name, alias, version):
            self.alias = version

    class Ledger:
        latest = None
        def save(self, value):
            self.latest = value
        def latest_promoted(self, object_slug):
            return self.latest
        def previous_promoted(self, object_slug, release_id):
            return None

    registry, ledger = Registry(), Ledger()
    manager = ReleaseManager(registry, ledger)
    with pytest.raises(ValueError):
        manager.promote(release(), "")
    promoted = manager.promote(release(), "operator")
    assert promoted.state == "promoted"
    assert registry.alias == "2"
