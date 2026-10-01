"""Protected operator authority and admission for the single-VM executor.

Readiness is evidence supplied by the infrastructure/runtime owners. This module
does not certify an image, approve a dataset, or infer readiness from disk size.
"""
from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from defect_platform.catalog_store import ClassCatalogStore
from defect_platform.contracts import CertifiedRuntime, DatasetVersion, ExperimentConfig
from defect_platform.control.queue_contracts import (
    QueueIntent,
    QueueModel,
    QueueOperatorConfig,
    VMJobProfile,
)
from defect_platform.control.vm_executor import HostIdentity
from defect_platform.dataset import load_dataset_semantics
from defect_platform.semantics import SemanticManifest, fingerprint


class VMReadiness(QueueModel):
    project_id: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    instance_id: str = Field(pattern=r"^[0-9]+$")
    boot_id: str = Field(min_length=1)
    gpu_identity: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    toolkit_version: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    compatible_digests: list[str] = Field(min_length=1)
    observed_at: datetime
    expires_at: datetime
    evidence_ref: str = Field(min_length=1)
    metadata_denial_evidence_ref: str = Field(min_length=1)
    credential_scope_evidence_ref: str = Field(min_length=1)
    boot_disk_evidence_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def finite_window(self):
        if (self.observed_at.tzinfo is None or self.expires_at.tzinfo is None
                or self.expires_at <= self.observed_at):
            raise ValueError("readiness requires a finite timezone-aware evidence window")
        return self

    def validate_intent(self, intent: QueueIntent, now: datetime,
                        identity: HostIdentity | None = None) -> None:
        if not self.observed_at <= now < self.expires_at:
            raise ValueError("VM readiness evidence is stale or from the future")
        job = intent.job
        if (job.project_id, job.zone, job.instance_id, job.gpu_identity) != (
                self.project_id, self.zone, self.instance_id, self.gpu_identity):
            raise ValueError("job identity differs from approved VM/GPU readiness")
        if job.image_digest not in self.compatible_digests:
            raise ValueError("exact certified digest has no VM compatibility evidence")
        if identity is not None and identity != HostIdentity(
                self.project_id, self.zone, self.instance_id, self.boot_id):
            raise ValueError("observed VM/boot identity differs from readiness")


class QueueServiceConfig(QueueModel):
    """Only an operator-protected local file can supply these authorities."""

    scope: str = Field(min_length=1)
    storage_path: Path
    socket_path: Path
    catalog_root: str = Field(min_length=1)
    allowed_operator_uids: list[int] = Field(min_length=1)
    worker_uids: list[int] = Field(min_length=1)
    socket_group_id: int = Field(ge=0)
    state_group_id: int = Field(ge=0)
    tick_interval_seconds: float = Field(gt=0, le=60)
    worker_id: str = Field(min_length=1)
    lease_seconds: int = Field(gt=0)
    operator: QueueOperatorConfig
    readiness: VMReadiness
    approved_runtimes: dict[str, CertifiedRuntime] = Field(min_length=1)
    approved_datasets: dict[str, DatasetVersion] = Field(min_length=1)
    approved_experiments: dict[str, ExperimentConfig] = Field(min_length=1)
    approved_profiles: dict[str, VMJobProfile] = Field(min_length=1)
    experiment_profiles: dict[str, str] = Field(min_length=1)
    output_root_uri: str = Field(pattern=r"^gs://[^/\s]+/[^\s]+$")
    mlflow_tracking_uri: str = Field(pattern=r"^https://[^\s]+$")
    mlflow_experiment_name: str = Field(min_length=1)
    credential_path: Path
    trainer_uid: int = Field(gt=0)
    trainer_gid: int = Field(gt=0)
    gpu_devices: list[str] = Field(min_length=1)
    network_name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    memory_bytes: int = Field(gt=0)
    pids_limit: int = Field(gt=0)
    vm_cost_upper_bound_hourly_usd: float = Field(gt=0)
    cost_authorized_from: datetime
    cost_authorized_until: datetime
    backup_root_uri: str = Field(pattern=r"^gs://[^/\s]+/[^\s]+$")
    backup_interval_seconds: int = Field(gt=0)
    backup_local_keep: int = Field(gt=0)

    @model_validator(mode="after")
    def protected_layout(self):
        for path in (self.storage_path, self.socket_path, self.credential_path):
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("service paths must be absolute without parent traversal")
        if self.network_name in {"host", "container", "none", "bridge"}:
            raise ValueError("a separately verified isolated trainer network is required")
        if set(self.allowed_operator_uids) & set(self.worker_uids):
            raise ValueError("operator and worker UID authority must be distinct")
        if any(uid < 0 for uid in self.allowed_operator_uids + self.worker_uids):
            raise ValueError("principal UIDs cannot be negative")
        if self.trainer_uid in set(self.allowed_operator_uids + self.worker_uids):
            raise ValueError("trainer UID cannot access the queue service")
        if (self.cost_authorized_from.tzinfo is None or self.cost_authorized_until.tzinfo is None
                or self.cost_authorized_until <= self.cost_authorized_from):
            raise ValueError("a finite timezone-aware campaign cost window is required")
        for key, profile in self.approved_profiles.items():
            if key != profile.profile_id:
                raise ValueError("approved VM profile identity mismatch")
            bound = self.vm_cost_upper_bound_hourly_usd * profile.max_runtime_seconds / 3600
            if profile.per_run_cost_usd < bound:
                raise ValueError("per-run cost reservation must cover maximum VM runtime")
        if (self.operator.vm_hourly_cost_usd != self.vm_cost_upper_bound_hourly_usd
                or self.operator.cost_meter_started_at != self.cost_authorized_from):
            raise ValueError("persistent cost meter must match protected campaign authority")
        return self


def profile_fingerprint(profile: VMJobProfile) -> str:
    return fingerprint(profile.model_dump(mode="json"))


def require_protected_file(path: Path) -> None:
    """Reject symlinks and writable authority paths, including their parents."""
    if not path.is_absolute():
        raise ValueError("operator authority path must be absolute")
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_mode & 0o022:
            raise PermissionError(f"operator authority path is not protected: {item}")
        if info.st_uid not in {0, os.geteuid()}:
            raise PermissionError(f"operator authority path has an untrusted owner: {item}")
    if not path.is_file():
        raise ValueError("operator authority must be a regular file")


def load_queue_service_config(config_path: Path) -> QueueServiceConfig:
    require_protected_file(config_path)
    return QueueServiceConfig.model_validate(yaml.safe_load(config_path.read_text()))


class QueueAdmission:
    def __init__(self, config: QueueServiceConfig, catalogs: ClassCatalogStore,
                 *, semantic_loader=load_dataset_semantics, refresh=None):
        self.config, self.catalogs, self.semantic_loader = config, catalogs, semantic_loader
        self.refresh = refresh

    def validate(self, intent: QueueIntent, *, now: datetime,
                 identity: HostIdentity | None = None) -> SemanticManifest:
        intent = QueueIntent.model_validate(intent.model_dump(mode="json"))
        # The current protected authority is consulted again at delayed dispatch.
        config = self.refresh() if self.refresh else self.config
        if not config.cost_authorized_from <= now < config.cost_authorized_until:
            raise ValueError("campaign authorization is inactive or expired")
        for supplied, approved in (
                (intent.experiment, config.approved_experiments.get(intent.experiment.experiment_id)),
                (intent.runtime, config.approved_runtimes.get(intent.runtime.runtime_id)),
                (intent.dataset, config.approved_datasets.get(intent.dataset.version_id)),
                (intent.job, config.approved_profiles.get(intent.job.profile_id))):
            if approved is None or supplied.model_dump(mode="json") != approved.model_dump(mode="json"):
                raise ValueError("request differs from current protected experiment/runtime/dataset/profile authority")
        config.readiness.validate_intent(intent, now, identity)
        if config.experiment_profiles.get(intent.experiment.experiment_id) != intent.job.profile_id:
            raise ValueError("resource estimate/profile is not approved for this exact experiment")
        semantic = SemanticManifest.model_validate(
            self.semantic_loader(intent.dataset).model_dump(mode="json"))
        if (semantic.kind != "dataset" or semantic.sha256 != intent.dataset.semantic_sha256
                or semantic.dataset_version_id != intent.dataset.version_id
                or semantic.object_slug != intent.experiment.object_slug):
            raise ValueError("verified immutable dataset semantics do not match request")
        catalog = self.catalogs.get_catalog(intent.dataset.object_slug, semantic.catalog.sha256)
        if (catalog is None or catalog.review_status != "reviewed"
                or catalog.sha256 != semantic.catalog.sha256
                or catalog.object_slug != intent.experiment.object_slug
                or catalog.labels != intent.classes or semantic.catalog.labels != catalog.labels):
            raise ValueError("request classes differ from trusted reviewed catalog")
        if intent.experiment.catalog_sha256 != catalog.sha256:
            raise ValueError("queued experiment must pin the reviewed catalog fingerprint")
        # VM trainers receive GCS grants; arbitrary host-local dataset paths are denied.
        uris = [intent.dataset.root_uri, intent.dataset.manifest_uri,
                intent.dataset.source_snapshot_uri, intent.dataset.semantic_manifest_uri,
                intent.experiment.model.weights_uri,
                *(uri for values in intent.dataset.shard_uris.values() for uri in values)]
        if any(not uri or not uri.startswith("gs://") for uri in uris):
            raise ValueError("VM datasets and weights require immutable approved GCS references")
        if now.timestamp() + intent.job.max_runtime_seconds > config.cost_authorized_until.timestamp():
            raise ValueError("maximum run duration exceeds remaining campaign authorization")
        return semantic
