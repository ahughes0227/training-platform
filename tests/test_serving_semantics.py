import base64
import io
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from PIL import Image

from defect_platform.contracts import ModelRelease
from defect_platform.semantics import ClassCatalog, ClassDefinition
from defect_platform.serve.api import PredictionService, PredictRequest
from defect_platform.serve.deploy import rayservice_manifest
from defect_platform.serve.releases import ReleaseManager


def catalog():
    return ClassCatalog(catalog_id="part", object_slug="part", review_status="reviewed",
        reviewed_by="operator", reviewed_at=datetime.now(UTC), classes=[
            ClassDefinition(class_id="crack", label="crack", definition="Material separation"),
            ClassDefinition(class_id="dent", label="dent", definition="Surface deformation")])


def candidate():
    return ModelRelease(object_slug="part", model_name="defect-part", model_version="2",
        release_id="r-2", run_id="run-2", dataset_version_id="ds-1", runtime_id="rt-1",
        serving_image_digest="repo/serve@sha256:"+"a"*64, state="staged",
        catalog_sha256="b"*64, model_semantic_sha256="c"*64, bundle_sha256="d"*64)


@pytest.mark.parametrize("mutation", [
    {"class_id": "dent"}, {"class_name": "unknown"}, {"catalog_sha256": "f"*64},
    {"semantic_sha256": "f"*64}])
def test_serving_rejects_wrong_class_meaning_or_model_identity(mutation):
    value = catalog()
    output = {"class_name": "crack", "class_id": "crack", "catalog_sha256": value.sha256,
              "semantic_sha256": "c"*64, "confidence": 0.9, "review": False, **mutation}
    service = PredictionService(lambda image, explain: output, "defect-part", "2",
                                catalog=value, semantic_sha256="c"*64)
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8)).save(encoded, format="PNG")
    with pytest.raises(ValueError):
        service.predict(PredictRequest(image_base64=base64.b64encode(encoded.getvalue()).decode()))


def test_release_validation_failure_has_no_alias_ledger_or_deploy_effects():
    effects = []
    class Registry:
        def verify_release(self, release):
            raise ValueError("bundle does not match pinned fingerprint")
        def set_registered_model_alias(self, *args): effects.append("alias")
    ledger = SimpleNamespace(save=lambda value: effects.append("ledger"))
    manager = ReleaseManager(Registry(), ledger)
    with pytest.raises(ValueError, match="pinned fingerprint"):
        manager.promote(candidate(), "operator", deploy=lambda value: effects.append("deploy"))
    assert effects == []
    missing = candidate().model_copy(update={"bundle_sha256": None})
    with pytest.raises(ValueError, match="lacks pinned"):
        manager.promote(missing, "operator")
    assert effects == []


def test_rayservice_pins_exact_model_version_and_semantic_bundle():
    release = candidate().model_copy(update={"state": "promoted", "approved_by": "operator",
                                             "approved_at": datetime.now(UTC)})
    manifest = rayservice_manifest(release, mlflow_tracking_uri="https://mlflow",
                                   catalog_root="gs://catalogs/approved")
    env = {entry["name"]: entry["value"] for entry in manifest["spec"]["rayClusterConfig"]
           ["headGroupSpec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["DEFECT_MODEL_VERSION"] == "2"
    assert env["DEFECT_CATALOG_SHA256"] == release.catalog_sha256
    assert env["DEFECT_MODEL_SEMANTIC_SHA256"] == release.model_semantic_sha256
    assert env["DEFECT_BUNDLE_SHA256"] == release.bundle_sha256
