from __future__ import annotations

from types import SimpleNamespace

import pytest

from defect_platform.contracts import ModelRelease
from defect_platform.serve.ledger import GCSReleaseLedger, SqlReleaseLedger
from defect_platform.serve.releases import ReleaseManager


def candidate(version: str) -> ModelRelease:
    return ModelRelease(
        object_slug="valves", model_name="defect-valves", model_version=version,
        release_id=f"release-{version}", run_id=f"run-{version}",
        dataset_version_id="ds-1", runtime_id="runtime-1",
        serving_image_digest="registry/serve@sha256:" + "a" * 64,
        state="staged",
        catalog_sha256="b" * 64, model_semantic_sha256="c" * 64, bundle_sha256="d" * 64,
    )


class Registry:
    alias: str | None = None

    def verify_release(self, release):
        assert release.bundle_sha256 == "d" * 64

    def get_model_version(self, name, version):
        return SimpleNamespace(version=version)

    def get_model_version_by_alias(self, name, alias):
        if self.alias is None:
            error = RuntimeError("no alias")
            error.error_code = "RESOURCE_DOES_NOT_EXIST"
            raise error
        return SimpleNamespace(version=self.alias)

    def set_registered_model_alias(self, name, alias, version):
        self.alias = version

    def delete_registered_model_alias(self, name, alias):
        self.alias = None


def test_release_ledger_promotion_rollback_and_deploy_failure(tmp_path):
    ledger = SqlReleaseLedger(f"sqlite:///{tmp_path}/state.sqlite3")
    registry = Registry()
    manager = ReleaseManager(registry, ledger)
    first = manager.promote(candidate("1"), "alice", deploy=lambda release: None)
    assert ledger.latest_promoted("valves").release_id == first.release_id
    with pytest.raises(RuntimeError, match="deployment failed"):
        manager.promote(candidate("2"), "bob", deploy=lambda release: (_ for _ in ()).throw(RuntimeError("deployment failed")))
    assert registry.alias == "1"
    assert ledger.get("release-2").state == "staged"
    second = manager.promote(candidate("2"), "bob", deploy=lambda release: None)
    assert registry.alias == "2"
    assert second.approved_by == "bob"
    restored = manager.rollback("valves", "carol", deploy=lambda release: None)
    assert restored.release_id == first.release_id
    assert registry.alias == "1"
    assert ledger.get(second.release_id).state == "rolled_back"


def test_rollback_restores_alias_if_deploy_fails(tmp_path):
    ledger = SqlReleaseLedger(str(tmp_path / "state.sqlite3"))
    registry = Registry()
    manager = ReleaseManager(registry, ledger)
    manager.promote(candidate("1"), "alice")
    manager.promote(candidate("2"), "bob")
    with pytest.raises(RuntimeError):
        manager.rollback("valves", "carol", deploy=lambda release: (_ for _ in ()).throw(RuntimeError("failed")))
    assert registry.alias == "2"
    assert ledger.get("release-2").state == "promoted"


def test_gcs_release_history_and_rollback_without_cloud_sql():
    class Blob:
        def __init__(self, name, objects):
            self.name, self.objects = name, objects

        def upload_from_string(self, content, *, content_type, if_generation_match):
            assert content_type == "application/json" and if_generation_match == 0
            assert self.name not in self.objects
            self.objects[self.name] = content.encode()

        def download_as_bytes(self):
            return self.objects[self.name]

    class Storage:
        def __init__(self):
            self.objects = {}

        def bucket(self, name):
            assert name == "artifacts"
            return self

        def blob(self, name):
            return Blob(name, self.objects)

        def list_blobs(self, name, *, prefix):
            assert name == "artifacts"
            return [Blob(key, self.objects) for key in self.objects if key.startswith(prefix)]

    ledger = GCSReleaseLedger("gs://artifacts/releases", client=Storage())
    registry = Registry()
    manager = ReleaseManager(registry, ledger)
    first = manager.promote(candidate("1"), "alice")
    second = manager.promote(candidate("2"), "bob")
    assert ledger.latest_promoted("valves").release_id == second.release_id
    assert manager.rollback("valves", "carol").release_id == first.release_id
    assert ledger.get(second.release_id).state == "rolled_back"
    assert registry.alias == "1"


def test_release_commands_use_remote_run_and_gcs_ledger(monkeypatch):
    from defect_platform.serve import commands

    monkeypatch.setenv("DEFECT_CONTROL_SERVICE_URL", "https://control.run.app")
    monkeypatch.setenv("DEFECT_RELEASE_GCS_URI", "gs://artifacts/releases")
    monkeypatch.setattr(commands, "ControlAPIClient", lambda url: SimpleNamespace(get=lambda run_id: run_id))
    monkeypatch.setattr(commands, "GCSReleaseLedger", lambda uri: uri)
    assert commands._run("run-1") == "run-1"
    assert commands._ledger() == "gs://artifacts/releases"
