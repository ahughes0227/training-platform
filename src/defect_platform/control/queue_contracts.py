"""Typed records for the single-VM durable training queue.

These are control-plane records, not trainer or runtime packaging contracts.
An operator must supply measured limits and readiness; this module invents none.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import ConfigDict, Field, model_validator

from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetVersion,
    ExperimentConfig,
    StrictModel,
)


class QueueModel(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class QueueState(StrEnum):
    WAITING = "waiting"
    DISPATCHING = "dispatching"
    ACTIVE = "active"
    BLOCKED = "blocked"
    CANCELING = "canceling"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


class QueuePolicyState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"


class FailurePolicy(StrEnum):
    PAUSE = "pause"
    CONTINUE = "continue"


class RetryClassification(StrEnum):
    TRANSIENT = "transient"
    INTERRUPTED = "interrupted"
    DETERMINISTIC = "deterministic"
    UNKNOWN = "unknown"


class ExternalOutcome(StrEnum):
    UNKNOWN = "unknown"
    CREATE_INTENT = "create_intent"
    CREATED = "created"
    START_INTENT = "start_intent"
    STARTED = "started"
    TERMINAL = "terminal"


class AttemptState(StrEnum):
    DISPATCHING = "dispatching"
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    SKIPPED = "skipped"
    TERMINAL = "terminal"


class VMJobProfile(QueueModel):
    """Immutable per-request VM/container and resource/cost estimate."""

    profile_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    gpu_identity: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^[^\s]+@sha256:[a-f0-9]{64}$")
    estimated_runtime_seconds: int = Field(gt=0)
    max_runtime_seconds: int = Field(gt=0)
    per_run_cost_usd: float = Field(gt=0)
    storage_peak_bytes: int = Field(gt=0)
    storage_peak_inodes: int = Field(gt=0)
    max_attempts: int = Field(ge=1)

    @model_validator(mode="after")
    def runtime_bound(self) -> VMJobProfile:
        if self.max_runtime_seconds < self.estimated_runtime_seconds:
            raise ValueError("maximum runtime must cover the estimate")
        return self

    @property
    def profile_sha256(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QueueIntent(QueueModel):
    """Prepared, authority-bound request accepted by the queue."""

    experiment: ExperimentConfig
    dataset: DatasetVersion
    runtime: CertifiedRuntime
    classes: list[str] = Field(min_length=2)
    job: VMJobProfile
    idempotency_key: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def exact_bindings(self) -> QueueIntent:
        if not self.runtime.certified or self.runtime.image_digest != self.job.image_digest:
            raise ValueError("queue intent requires the exact certified runtime image digest")
        if self.experiment.dataset_version_id != self.dataset.version_id:
            raise ValueError("experiment must bind the supplied immutable dataset version")
        if self.experiment.runtime_id != self.runtime.runtime_id:
            raise ValueError("experiment must bind the supplied certified runtime")
        if self.experiment.object_slug != self.dataset.object_slug:
            raise ValueError("experiment and dataset object identities must match")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("canonical class names must be unique")
        return self

    @property
    def config_sha256(self) -> str:
        """Fingerprint canonical experiment and selected immutable identities."""
        payload = {
            "experiment": self.experiment.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "runtime": self.runtime.model_dump(mode="json"),
            "classes": self.classes,
            "job": self.job.model_dump(mode="json"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class QueueOperatorConfig(QueueModel):
    """Required operator-supplied queue limits, measured readiness and approvals."""

    revision: int = Field(ge=1)
    max_backlog: int = Field(gt=0)
    campaign_cost_cap_usd: float = Field(gt=0)
    campaign_authorization_id: str = Field(min_length=1)
    cost_meter_started_at: datetime
    vm_hourly_cost_usd: float = Field(gt=0)
    reserved_headroom_bytes: int = Field(gt=0)
    reserved_headroom_inodes: int = Field(gt=0)
    usable_capacity_bytes: int = Field(gt=0)
    observed_free_bytes: int = Field(ge=0)
    observed_free_inodes: int = Field(ge=0)
    capacity_observed_at: datetime
    max_start_delay_seconds: int = Field(gt=0)
    stalled_work_seconds: int = Field(gt=0)
    recovery_bound_seconds: int = Field(gt=0)
    retry_backoff_seconds: int = Field(ge=0)
    max_capacity_observation_age_seconds: int = Field(gt=0)
    failure_policy: FailurePolicy = FailurePolicy.PAUSE
    backup_evidence_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def measured_capacity_consistent(self) -> QueueOperatorConfig:
        if self.observed_free_bytes > self.usable_capacity_bytes:
            raise ValueError("observed free bytes cannot exceed measured usable capacity")
        return self


class QueueEntry(QueueModel):
    entry_id: str
    run_id: str
    scope: str
    sequence: int = Field(gt=0)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    state: QueueState
    revision: int = Field(ge=1)
    intent: QueueIntent
    budget_reserved_usd: float = Field(ge=0)
    storage_reserved_bytes: int = Field(ge=0)
    storage_reserved_inodes: int = Field(ge=0)
    active_attempt_id: str | None = None
    blocking_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    last_observation_at: datetime | None = None
    container_exit_code: int | None = None
    outputs_verified: bool | None = None
    mlflow_finalized: bool | None = None
    release_eligible: bool = False
    cancellation_requested: bool = False
    next_eligible_at: datetime | None = None


class QueueAttempt(QueueModel):
    attempt_id: str
    entry_id: str
    run_id: str
    dispatch_id: str
    state: AttemptState
    external_outcome: ExternalOutcome
    revision: int = Field(ge=1)
    worker_id: str
    claim_token: str
    lease_expires_at: datetime
    project_id: str
    zone: str
    instance_id: str
    boot_id: str
    container_name: str
    container_id: str | None = None
    process_id: str | None = None
    exit_code: int | None = None
    log_ref: str | None = None
    output_ref: str | None = None
    started_at: datetime | None = None
    start_intent_at: datetime | None = None
    deadline_at: datetime | None = None
    budget_reserved_usd: float = Field(ge=0)
    storage_reserved_bytes: int = Field(ge=0)
    storage_reserved_inodes: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class QueueEvent(QueueModel):
    event_id: str
    scope: str
    entry_id: str | None = None
    attempt_id: str | None = None
    actor: str
    at: datetime
    action: str
    prior_revision: int | None = None
    resulting_revision: int
    configuration_id: str | None = None
    runtime_digest: str | None = None
    vm_identity: str | None = None
    container_identity: str | None = None
    process_identity: str | None = None
    log_ref: str | None = None
    evidence_ref: str | None = None
    detail: str | None = None


class QueueSnapshot(QueueModel):
    schema_version: int = Field(ge=1)
    scope: str
    policy_state: QueuePolicyState
    policy_revision: int = Field(ge=1)
    operator_config: QueueOperatorConfig
    entries: list[QueueEntry]
    attempts: list[QueueAttempt]
    events: list[QueueEvent]


class QueueClaim(QueueModel):
    entry: QueueEntry
    attempt: QueueAttempt


class QueueStore(Protocol):
    """Transactional persistence API; implementations must use CAS and atomic admission."""

    def initialize(self, *, scope: str, operator_config: QueueOperatorConfig,
                   actor: str, now: datetime) -> None: ...
    def policy(self, scope: str) -> dict: ...

    def enqueue_batch(self, *, scope: str, intents: list[QueueIntent],
                      batch_idempotency_key: str,
                      operator_config: QueueOperatorConfig,
                      expected_policy_revision: int, actor: str,
                      observed_free_bytes: int, observed_free_inodes: int,
                      capacity_evidence_ref: str, capacity_observed_at: datetime,
                      now: datetime) -> list[QueueEntry]: ...
    def get_entry(self, entry_id: str) -> QueueEntry | None: ...
    def list_entries(self, scope: str) -> list[QueueEntry]: ...
    def get_attempt(self, attempt_id: str) -> QueueAttempt | None: ...
    def list_attempts(self, entry_id: str) -> list[QueueAttempt]: ...
    def list_events(self, scope: str, entry_id: str | None = None) -> list[QueueEvent]: ...
    def claim_next(self, *, scope: str, worker_id: str, lease_seconds: int,
                   boot_id: str, observed_free_bytes: int, observed_free_inodes: int,
                   capacity_evidence_ref: str, capacity_observed_at: datetime,
                   now: datetime) -> QueueClaim | None: ...
    def record_dispatch(self, *, entry_id: str, expected_revision: int,
                        attempt_id: str, dispatch_id: str,
                        claim_token: str,
                        external_outcome: ExternalOutcome, actor: str,
                        now: datetime, container_id: str | None = None,
                        process_id: str | None = None, exit_code: int | None = None,
                        log_ref: str | None = None, output_ref: str | None = None,
                        evidence_ref: str | None = None) -> QueueEntry: ...
    def transition(self, *, entry_id: str, expected_revision: int, state: QueueState,
                   actor: str, action: str, now: datetime,
                   blocking_reason: str | None = None,
                   next_eligible_at: datetime | None = None,
                   evidence_ref: str | None = None) -> QueueEntry: ...
    def set_policy(self, *, scope: str, expected_policy_revision: int,
                   state: QueuePolicyState, actor: str, now: datetime) -> int: ...
    def cancel_waiting(self, *, entry_id: str, expected_revision: int,
                       actor: str, now: datetime) -> QueueEntry: ...
    def request_active_cancel(self, *, entry_id: str, expected_revision: int,
                              actor: str, now: datetime) -> QueueEntry: ...
    def retry(self, *, entry_id: str, expected_revision: int, actor: str,
              now: datetime, evidence_ref: str,
              classification: RetryClassification = RetryClassification.UNKNOWN) -> QueueEntry: ...
    def mark_interrupted(self, *, entry_id: str, expected_revision: int,
                         actor: str, now: datetime, evidence_ref: str) -> QueueEntry: ...
    def skip(self, *, entry_id: str, expected_revision: int, actor: str,
             now: datetime, evidence_ref: str) -> QueueEntry: ...
    def reconcile_unknown(self, *, entry_id: str, expected_revision: int,
                          resolution: ExternalOutcome, actor: str, now: datetime,
                          evidence_ref: str) -> QueueEntry: ...
    def record_finalization(self, *, entry_id: str, expected_revision: int,
                            outputs_verified: bool, mlflow_finalized: bool,
                            actor: str, now: datetime, evidence_ref: str) -> QueueEntry: ...
    def record_exposure(self, *, scope: str, expected_policy_revision: int,
                        additional_cost_usd: float, actor: str, now: datetime,
                        evidence_ref: str) -> float: ...
    def account_until(self, *, scope: str, expected_policy_revision: int,
                      observed_at: datetime, actor: str, evidence_ref: str) -> float: ...
    def backup(self, destination: str) -> str: ...
    def verify_restore(self, source: str) -> QueueSnapshot: ...
