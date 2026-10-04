"""Explicit approval, MLflow alias switching, and rollback for served models."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from defect_platform.contracts import ModelRelease


class RegistryClient(Protocol):
    def get_model_version_by_alias(self, name: str, alias: str): ...
    def get_model_version(self, name: str, version: str): ...
    def set_registered_model_alias(self, name: str, alias: str, version: str): ...
    def delete_registered_model_alias(self, name: str, alias: str): ...
    def verify_release(self, release: ModelRelease) -> None: ...


class ReleaseLedger(Protocol):
    def save(self, release: ModelRelease) -> None: ...
    def get(self, release_id: str) -> ModelRelease: ...
    def latest_promoted(self, object_slug: str) -> ModelRelease | None: ...
    def previous_promoted(self, object_slug: str, release_id: str) -> ModelRelease | None: ...


class ReleaseManager:
    """Only explicit approver calls may move the live model alias."""

    def __init__(self, registry: RegistryClient, ledger: ReleaseLedger):
        self.registry = registry
        self.ledger = ledger

    def promote(
        self,
        candidate: ModelRelease,
        approver: str,
        deploy: Callable[[ModelRelease], None] | None = None,
    ) -> ModelRelease:
        if not approver.strip():
            raise ValueError("approver identity is required")
        if candidate.state != "staged":
            raise ValueError("only a staged release can be promoted")
        if "@sha256:" not in candidate.serving_image_digest:
            raise ValueError("serving image must use an immutable digest")
        if not all(
            (candidate.catalog_sha256, candidate.model_semantic_sha256, candidate.bundle_sha256)
        ):
            raise ValueError("release lacks pinned semantic identity and bundle integrity")
        self.registry.verify_release(candidate)
        version = self.registry.get_model_version(candidate.model_name, candidate.model_version)
        if version is None or not hasattr(version, "version"):
            raise ValueError("MLflow model version could not be resolved")
        if str(version.version) != candidate.model_version:
            raise ValueError("MLflow model version does not match candidate")
        approved = candidate.model_copy(
            update={
                "approved_by": approver,
                "approved_at": datetime.now(UTC),
            }
        )
        # Record intent first. A failed alias switch leaves a staged release;
        # a failed final ledger write can be reconciled against the alias.
        self.ledger.save(approved)
        promoted = approved.model_copy(update={"state": "promoted"})
        try:
            previous = self.registry.get_model_version_by_alias(candidate.model_name, "champion")
        except AttributeError:
            previous = None
        except Exception as exc:
            if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
                raise
            previous = None
        self.registry.set_registered_model_alias(
            candidate.model_name, "champion", candidate.model_version
        )
        try:
            if deploy is not None:
                deploy(promoted)
        except Exception:
            if previous is None:
                self.registry.delete_registered_model_alias(candidate.model_name, "champion")
            else:
                self.registry.set_registered_model_alias(
                    candidate.model_name, "champion", str(previous.version)
                )
            raise
        self.ledger.save(promoted)
        return promoted

    def rollback(
        self,
        object_slug: str,
        approver: str,
        deploy: Callable[[ModelRelease], None] | None = None,
    ) -> ModelRelease:
        if not approver.strip():
            raise ValueError("approver identity is required")
        current = self.ledger.latest_promoted(object_slug)
        if current is None:
            raise ValueError("no promoted model to roll back")
        previous = self.ledger.previous_promoted(object_slug, current.release_id)
        if previous is None:
            raise ValueError("no previous promoted model")
        if not all(
            (previous.catalog_sha256, previous.model_semantic_sha256, previous.bundle_sha256)
        ):
            raise ValueError(
                "previous release requires explicit semantic migration before rollback"
            )
        self.registry.verify_release(previous)
        self.registry.get_model_version(previous.model_name, previous.model_version)
        self.registry.set_registered_model_alias(
            previous.model_name, "champion", previous.model_version
        )
        try:
            if deploy is not None:
                deploy(previous)
        except Exception:
            self.registry.set_registered_model_alias(
                current.model_name, "champion", current.model_version
            )
            raise
        rolled_back = current.model_copy(update={"state": "rolled_back"})
        self.ledger.save(rolled_back)
        return previous


class MlflowRegistryAdapter:
    def __init__(self, tracking_uri: str):
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.client = MlflowClient()

    def bundle_identity(
        self, name: str, version: str, *, expected_release: ModelRelease | None = None
    ) -> dict[str, str]:
        from pathlib import Path

        import mlflow

        from defect_platform.catalog_store import catalog_store_from_env
        from defect_platform.mlflow_auth import mlflow_tracking_auth
        from defect_platform.semantics import read_manifest
        from defect_platform.trainer.weights import artifact_sha256, verify_bundle_integrity

        with mlflow_tracking_auth(self.tracking_uri):
            record = self.client.get_model_version(name, version)
            artifacts = getattr(mlflow, "artifacts", None)
            if artifacts is None:
                raise RuntimeError("installed MLflow does not expose artifact downloads")
            directory = Path(artifacts.download_artifacts(artifact_uri=record.source))
        verify_bundle_integrity(
            directory,
            expected_sha256=(expected_release.bundle_sha256 if expected_release else None),
        )
        manifest = read_manifest(
            directory / "semantics.json",
            expected_release.model_semantic_sha256 if expected_release else None,
        )
        if manifest.kind != "model":
            raise ValueError("release requires model semantic manifest")
        store = catalog_store_from_env()
        catalog = (
            store.get_catalog(manifest.object_slug, manifest.catalog.sha256) if store else None
        )
        if catalog is None or catalog.review_status != "reviewed":
            raise ValueError("release catalog is absent from operator approval store")
        from defect_platform.trainer.inference import load_inference_bundle

        load_inference_bundle(
            directory,
            expected_semantic_sha256=manifest.sha256,
            expected_catalog_sha256=catalog.sha256,
            expected_bundle_sha256=(expected_release.bundle_sha256 if expected_release else None),
        )
        required_runtime_digest = manifest.runtime_image_digest
        required_runtime_commit = manifest.runtime_source_commit
        required_runtime_id = manifest.runtime_id
        parent_semantic_sha256 = manifest.parent_semantic_sha256
        if (
            required_runtime_digest is None
            or required_runtime_commit is None
            or required_runtime_id is None
            or parent_semantic_sha256 is None
        ):
            raise ValueError("model semantic manifest lacks complete runtime and dataset lineage")
        return {
            "catalog_sha256": manifest.catalog.sha256,
            "model_semantic_sha256": manifest.sha256,
            "bundle_sha256": artifact_sha256(directory / "integrity.json"),
            "object_slug": manifest.object_slug,
            "dataset_version_id": manifest.dataset_version_id,
            "runtime_id": required_runtime_id,
            "runtime_image_digest": required_runtime_digest,
            "runtime_source_commit": required_runtime_commit,
            "dataset_semantic_sha256": parent_semantic_sha256,
        }

    def verify_release(self, release: ModelRelease) -> None:
        identity = self.bundle_identity(
            release.model_name, release.model_version, expected_release=release
        )
        for key in (
            "catalog_sha256",
            "model_semantic_sha256",
            "bundle_sha256",
            "object_slug",
            "dataset_version_id",
            "runtime_id",
        ):
            if identity[key] != getattr(release, key):
                raise ValueError(f"release {key} differs from verified model bundle")

    def __getattr__(self, name: str):
        return self._authenticated(name)

    def _authenticated(self, name: str):
        from defect_platform.mlflow_auth import mlflow_tracking_auth

        method = getattr(self.client, name)

        def authenticated(*args: Any, **kwargs: Any) -> Any:
            with mlflow_tracking_auth(self.tracking_uri):
                return method(*args, **kwargs)

        return authenticated

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        return self._authenticated("get_model_version_by_alias")(name, alias)

    def get_model_version(self, name: str, version: str) -> Any:
        return self._authenticated("get_model_version")(name, version)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> Any:
        return self._authenticated("set_registered_model_alias")(name, alias, version)

    def delete_registered_model_alias(self, name: str, alias: str) -> Any:
        return self._authenticated("delete_registered_model_alias")(name, alias)
