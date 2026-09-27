"""Explicit approval, MLflow alias switching, and rollback for served models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol

from defect_platform.contracts import ModelRelease


class RegistryClient(Protocol):
    def get_model_version_by_alias(self, name: str, alias: str): ...
    def get_model_version(self, name: str, version: str): ...
    def set_registered_model_alias(self, name: str, alias: str, version: str): ...
    def delete_registered_model_alias(self, name: str, alias: str): ...


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
        version = self.registry.get_model_version(candidate.model_name, candidate.model_version)
        if str(version.version) != candidate.model_version:
            raise ValueError("MLflow model version does not match candidate")
        approved = candidate.model_copy(
            update={
                "approved_by": approver,
                "approved_at": datetime.now(timezone.utc),
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
        self.registry.set_registered_model_alias(candidate.model_name, "champion", candidate.model_version)
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
        self.registry.get_model_version(previous.model_name, previous.model_version)
        self.registry.set_registered_model_alias(previous.model_name, "champion", previous.model_version)
        try:
            if deploy is not None:
                deploy(previous)
        except Exception:
            self.registry.set_registered_model_alias(current.model_name, "champion", current.model_version)
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

    def __getattr__(self, name: str):
        from defect_platform.mlflow_auth import mlflow_tracking_auth

        method = getattr(self.client, name)
        def authenticated(*args, **kwargs):
            with mlflow_tracking_auth(self.tracking_uri):
                return method(*args, **kwargs)
        return authenticated
