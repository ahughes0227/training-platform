# ruff: noqa: BLE001 -- every external boundary is converted into a held/unknown result.
"""Supervised single-VM queue worker with conservative engine reconciliation."""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from defect_platform.control.queue_contracts import (
    ExternalOutcome,
    QueueAttempt,
    QueueClaim,
    QueueEntry,
    QueueState,
    QueueStore,
)
from defect_platform.control.vm_executor import (
    ContainerSpec,
    ExecutionObservation,
    ExecutorError,
    HostIdentity,
    VMExecutor,
)


class WorkerAuthority(Protocol):
    """Integration seam for admission and trainer-output verification."""
    def validate_executor(self, executor: VMExecutor, identity: HostIdentity) -> None: ...
    def prepare_execution(self, entry: QueueEntry, attempt: QueueAttempt,
                          identity: HostIdentity) -> ContainerSpec: ...
    def read_prepared_execution(self, entry: QueueEntry,
                                attempt: QueueAttempt) -> ContainerSpec: ...
    def cleanup_verified_attempt(self, entry: QueueEntry, attempt: QueueAttempt,
                                 spec: ContainerSpec) -> None: ...
    def verify_outputs(self, entry: QueueEntry, attempt: QueueAttempt,
                       spec: ContainerSpec) -> dict: ...
    storage_path: Path
    def record_capacity(self, capacity, observed_at: datetime) -> str: ...
    def record_exposure(self, now: datetime) -> None: ...
    def maintain_backup(self, now: datetime) -> str: ...
    def heartbeat(self, result: WorkerTick) -> None: ...
    def persist_logs(self, entry: QueueEntry, attempt: QueueAttempt, text: str) -> str: ...
    def persist_worker_evidence(self, entry: QueueEntry, attempt: QueueAttempt,
                                kind: str, payload: dict) -> str: ...
    def record_failure(self, entry: QueueEntry, attempt: QueueAttempt,
                       spec: ContainerSpec | None,
                       observation: ExecutionObservation) -> dict: ...


@dataclass(frozen=True)
class WorkerTick:
    action: str
    entry_id: str | None = None
    attempt_id: str | None = None
    reason: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


class QueueWorker:
    """One-shot tick intended to run under a host supervisor, independent of SSH.

    The flock is held across reconciliation and engine operations, preventing
    concurrent local workers from racing create/start. Durable dispatch intents
    remain authoritative after crashes; lease expiry alone never replays work.
    """

    def __init__(self, *, store: QueueStore, executor: VMExecutor, authority: WorkerAuthority,
                 scope: str, worker_id: str, lock_path: Path, lease_seconds: int = 120,
                 clock: Callable[[], datetime] = _now):
        if not scope or not worker_id or lease_seconds < 1:
            raise ValueError("scope, worker_id, and positive lease_seconds are required")
        if not lock_path.is_absolute():
            raise ValueError("VM worker lock path must be absolute")
        self.store, self.executor, self.authority = store, executor, authority
        self.scope, self.worker_id, self.lock_path = scope, worker_id, lock_path
        self.lease_seconds, self.clock = lease_seconds, clock

    def tick(self) -> WorkerTick:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return WorkerTick("held", reason="another worker holds the VM dispatch lock")
            result = self._tick_locked()
            try:
                self.authority.heartbeat(result)
            except Exception as exc:
                return WorkerTick("held", result.entry_id, result.attempt_id,
                                  f"worker result was not durably heartbeated: {exc}; next tick will reconcile")
            return result
        finally:
            os.close(fd)

    def _tick_locked(self) -> WorkerTick:
        try:
            identity = self.executor.assert_host_identity()
        except Exception as exc:
            return WorkerTick("held", reason=f"VM/boot identity fence failed: {exc}")

        # Reconcile all persisted active/dispatching claims before considering new work.
        entries = self.store.list_entries(self.scope)
        unresolved: list[tuple[QueueEntry, QueueAttempt]] = []
        for entry in entries:
            if entry.state not in {QueueState.DISPATCHING, QueueState.ACTIVE,
                                   QueueState.CANCELING, QueueState.BLOCKED}:
                continue
            if not entry.active_attempt_id:
                return WorkerTick("held", entry.entry_id, reason="active queue state has no attempt identity")
            attempt = self.store.get_attempt(entry.active_attempt_id)
            if attempt is None:
                return WorkerTick("held", entry.entry_id, entry.active_attempt_id,
                                  "active queue state references a missing attempt")
            if not self._same_host(attempt, identity):
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  "persisted VM or boot identity differs; interrupted work requires operator reconciliation")
            unresolved.append((entry, attempt))
        if len(unresolved) > 1:
            return WorkerTick("held", reason="multiple unresolved active attempts violate single-slot policy")

        # Any GPU process or container outside the one exact persisted attempt blocks dispatch.
        scan = getattr(self.executor, "scan_gpu_workloads", None)
        if scan is None:
            return WorkerTick("held", reason="executor cannot prove absence of orphan GPU workloads")
        try:
            workloads = scan()
        except Exception as exc:
            return WorkerTick("held", reason=f"GPU workload observation failed: {exc}")
        allowed_id = unresolved[0][1].container_id if unresolved else None
        unknown = [w for w in workloads if w.get("container_id") != allowed_id]
        if unknown:
            return WorkerTick("held", reason="unbound container or GPU process detected; operator identity resolution required")

        if unresolved:
            entry, attempt = unresolved[0]
            result = self._reconcile(entry, attempt, identity)
            # Maintenance must never delay deadline cancellation or terminal
            # observation. Failures are retained as diagnostics after reconciliation.
            maintenance_errors = self._maintain()
            if maintenance_errors and result.action in {"active", "started", "unknown", "canceling"}:
                return WorkerTick(result.action, result.entry_id, result.attempt_id,
                                  f"{result.reason or 'existing work reconciled'}; {maintenance_errors}")
            return result
        maintenance_errors = self._maintain()
        if maintenance_errors:
            return WorkerTick("held", reason=f"new dispatch held because {maintenance_errors}")

        try:
            measured_at = self.clock()
            capacity = self.executor.disk_capacity(self.authority.storage_path)
            capacity_ref = self.authority.record_capacity(capacity, measured_at)
            claim = self.store.claim_next(
                scope=self.scope, worker_id=self.worker_id, lease_seconds=self.lease_seconds,
                boot_id=identity.boot_id, observed_free_bytes=capacity.free_bytes,
                observed_free_inodes=capacity.free_inodes,
                capacity_evidence_ref=capacity_ref, capacity_observed_at=measured_at,
                now=self.clock())
        except Exception as exc:
            return WorkerTick("held", reason=f"capacity admission evidence or claim failed: {exc}")
        if claim is None:
            return WorkerTick("idle")
        return self._dispatch(claim, identity)

    def _maintain(self) -> str | None:
        errors = []
        for name, operation in (("VM exposure", self.authority.record_exposure),
                                ("queue backup", self.authority.maintain_backup)):
            try:
                operation(self.clock())
            except Exception as exc:
                errors.append(f"{name} evidence failed: {exc}")
        return "; ".join(errors) if errors else None

    @staticmethod
    def _same_host(attempt: QueueAttempt, identity: HostIdentity) -> bool:
        return (attempt.project_id == identity.project and attempt.zone == identity.zone
                and attempt.instance_id == identity.instance_id and attempt.boot_id == identity.boot_id)

    @staticmethod
    def _labels_match(observation: ExecutionObservation, attempt: QueueAttempt) -> bool:
        labels = observation.labels
        return (labels.get("defect-platform.managed") == "true"
                and labels.get("defect-platform.run-id") == attempt.run_id
                and labels.get("defect-platform.attempt-id") == attempt.attempt_id)

    def _record(self, entry: QueueEntry, attempt: QueueAttempt, outcome: ExternalOutcome,
                **evidence) -> QueueEntry:
        return self.store.record_dispatch(
            entry_id=entry.entry_id, expected_revision=entry.revision,
            attempt_id=attempt.attempt_id, dispatch_id=attempt.dispatch_id,
            claim_token=attempt.claim_token,
            external_outcome=outcome, actor=self.worker_id, now=self.clock(), **evidence)

    def _dispatch(self, claim: QueueClaim, identity: HostIdentity) -> WorkerTick:
        entry, attempt = claim.entry, claim.attempt
        spec = None
        try:
            self.authority.validate_executor(self.executor, identity)
            spec = self.authority.prepare_execution(entry, attempt, identity)
            if spec.run_id != attempt.run_id or spec.attempt_id != attempt.attempt_id:
                raise ValueError("prepared execution identity does not match claimed attempt")
            if spec.image_digest != entry.intent.runtime.image_digest:
                raise ValueError("prepared image digest differs from the exact certified request digest")
            if self.executor.container_name(spec.run_id, spec.attempt_id) != attempt.container_name:
                raise ValueError("deterministic container name differs from persisted attempt identity")
        except Exception as exc:
            self._block(entry, f"dispatch admission failed: {exc}")
            return WorkerTick("blocked", entry.entry_id, attempt.attempt_id, str(exc))

        # Persist before the external side effect. Once this has happened, a new
        # worker inspects the deterministic name and never blindly repeats create.
        try:
            self._record(entry, attempt, ExternalOutcome.CREATE_INTENT)
            current = self.store.get_entry(entry.entry_id) or entry
            result = self.executor.create(spec)
            if (result.observation.state != "created" or not result.container_id
                    or result.observation.image != entry.intent.runtime.image_digest
                    or not self._labels_match(result.observation, attempt)):
                evidence = self.authority.persist_worker_evidence(entry, attempt, "create-unknown", {
                    "command": list(result.command), "detail": result.detail,
                    "observed_state": result.observation.state,
                    "observed_image": result.observation.image,
                    "observed_labels": result.observation.labels,
                })
                self._record(current, attempt, ExternalOutcome.UNKNOWN,
                             evidence_ref=evidence)
                return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                                  result.detail or "create result lacks exact created container identity")
            create_evidence = self.authority.persist_worker_evidence(entry, attempt, "created", {
                "command": list(result.command), "container_id": result.container_id,
                "image_digest": spec.image_digest, "detail": result.detail,
            })
            current = self._record(current, attempt, ExternalOutcome.CREATED,
                                   container_id=result.container_id, evidence_ref=create_evidence)
            attempt = self.store.get_attempt(attempt.attempt_id) or attempt
            # Persist start intent separately. Crash after start is recovered by inspect.
            self._record(current, attempt, ExternalOutcome.START_INTENT)
            current = self.store.get_entry(entry.entry_id) or current
            started = self.executor.start(result.container_id, run_id=attempt.run_id,
                                          attempt_id=attempt.attempt_id)
            if (started.observation.state not in {"running", "exited"}
                    or started.observation.image != entry.intent.runtime.image_digest
                    or not self._labels_match(started.observation, attempt)):
                start_evidence = self.authority.persist_worker_evidence(entry, attempt, "start-unknown", {
                    "command": list(started.command), "detail": started.detail,
                    "observed_state": started.observation.state,
                    "observed_image": started.observation.image,
                    "observed_labels": started.observation.labels,
                })
                latest = self.store.get_entry(entry.entry_id) or current
                self._record(latest, attempt, ExternalOutcome.UNKNOWN,
                             evidence_ref=start_evidence)
                return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                                  started.detail or "start outcome unresolved")
            latest = self.store.get_entry(entry.entry_id) or current
            self._record(latest, attempt, ExternalOutcome.STARTED,
                         container_id=started.container_id,
                         process_id=started.observation.process_id,
                         exit_code=started.observation.exit_code,
                         evidence_ref=self._evidence("started", started.detail))
            if started.observation.state == "exited":
                return self._finish(latest, attempt, spec, started.observation)
            return WorkerTick("started", entry.entry_id, attempt.attempt_id)
        except Exception as exc:
            # A subprocess error after intent is ambiguous. Keep the reservation and
            # let the next tick reconcile rather than classifying absence as failure.
            evidence_ref = None
            evidence_error = None
            try:
                evidence_ref = self.authority.persist_worker_evidence(entry, attempt,
                    "dispatch-error", {
                        "command": list(getattr(exc, "argv", ())),
                        "config_sha256": entry.intent.config_sha256,
                        "image_digest": entry.intent.runtime.image_digest,
                        "container_name": attempt.container_name,
                        "error": str(exc), "detail": getattr(exc, "detail", ""),
                    })
            except Exception as persist_exc:
                evidence_error = str(persist_exc)
            state_error = None
            try:
                latest = self.store.get_entry(entry.entry_id)
                current_attempt = self.store.get_attempt(attempt.attempt_id)
                if (latest is not None and current_attempt is not None
                        and latest.active_attempt_id == attempt.attempt_id
                        and current_attempt.external_outcome not in {
                            ExternalOutcome.UNKNOWN, ExternalOutcome.TERMINAL}):
                    self._record(latest, current_attempt, ExternalOutcome.UNKNOWN,
                                 evidence_ref=evidence_ref)
            except Exception as state_exc:
                state_error = str(state_exc)
            details = [f"dispatch outcome needs reconciliation: {exc}",
                       f"evidence={evidence_ref or 'unavailable'}"]
            if evidence_error:
                details.append(f"evidence write failed: {evidence_error}")
            if state_error:
                details.append(f"unknown-state write failed: {state_error}")
            return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                              "; ".join(details))

    def _reconcile(self, entry: QueueEntry, attempt: QueueAttempt,
                   identity: HostIdentity) -> WorkerTick:
        if self.executor.container_name(attempt.run_id, attempt.attempt_id) != attempt.container_name:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              "persisted container name does not match deterministic run/attempt identity")
        try:
            observation = self.executor.inspect(attempt.run_id, attempt.attempt_id,
                                                attempt.container_id)
        except Exception as exc:
            return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                              f"container inspection failed: {exc}")
        if observation.state in {"unknown", "absent"}:
            return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                              observation.error or "container outcome is unresolved; no replay")
        if not self._labels_match(observation, attempt):
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              "observed container labels do not match the persisted run and attempt")
        if observation.image != entry.intent.runtime.image_digest:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              "observed container image does not match the certified digest")
        if observation.state == "created":
            if entry.cancellation_requested:
                return self._cancel_unstarted(entry, attempt, observation)
            if attempt.external_outcome not in {
                    ExternalOutcome.CREATE_INTENT, ExternalOutcome.UNKNOWN,
                    ExternalOutcome.CREATED, ExternalOutcome.START_INTENT}:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  "created container has no persisted create/start intent")
            gate_error = self._maintain()
            if gate_error:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  f"created container start held because {gate_error}")
            try:
                self.authority.validate_executor(self.executor, identity)
                if attempt.external_outcome in {ExternalOutcome.CREATE_INTENT, ExternalOutcome.UNKNOWN}:
                    ref = self.authority.persist_worker_evidence(entry, attempt, "create-reconciled", {
                        "command": [self.executor.engine, "inspect", attempt.container_name],
                        "observed_state": observation.state,
                        "observed_image": observation.image,
                        "observed_labels": observation.labels,
                    })
                    self._record(entry, attempt, ExternalOutcome.CREATED,
                                 container_id=observation.container_id, evidence_ref=ref)
                    entry = self.store.get_entry(entry.entry_id) or entry
                    attempt = self.store.get_attempt(attempt.attempt_id) or attempt
                if attempt.external_outcome == ExternalOutcome.CREATED:
                    self._record(entry, attempt, ExternalOutcome.START_INTENT,
                                 evidence_ref=self._evidence("start-intent-reconciled", "created container observed"))
                    entry = self.store.get_entry(entry.entry_id) or entry
                    attempt = self.store.get_attempt(attempt.attempt_id) or attempt
                if attempt.external_outcome != ExternalOutcome.START_INTENT:
                    return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                      "persisted external state does not authorize container start")
                spec = self.authority.prepare_execution(entry, attempt, identity)
                result = self.executor.start(observation.container_id or attempt.container_id or "",
                                             run_id=attempt.run_id, attempt_id=attempt.attempt_id)
                if (result.observation.state == "running"
                        and result.observation.image == entry.intent.runtime.image_digest
                        and self._labels_match(result.observation, attempt)):
                    self._record(entry, attempt, ExternalOutcome.STARTED,
                                 container_id=result.container_id,
                                 process_id=result.observation.process_id)
                    return WorkerTick("started", entry.entry_id, attempt.attempt_id)
                if result.observation.state == "exited":
                    return self._finish(entry, attempt, spec, result.observation)
                return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                                  "start remains unresolved after reconciliation")
            except Exception as exc:
                return WorkerTick("unknown", entry.entry_id, attempt.attempt_id, str(exc))
        if observation.state == "running":
            if attempt.external_outcome != ExternalOutcome.STARTED:
                try:
                    self._record(entry, attempt, ExternalOutcome.STARTED,
                                 container_id=observation.container_id,
                                 process_id=observation.process_id,
                                 evidence_ref=self._evidence("reconciled-running", "exact labeled container observed"))
                    entry = self.store.get_entry(entry.entry_id) or entry
                    attempt = self.store.get_attempt(attempt.attempt_id) or attempt
                except Exception as exc:
                    return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                                      f"running container observed but could not reconcile start outcome: {exc}")
            if (entry.state == QueueState.ACTIVE and attempt.deadline_at is not None
                    and self.clock() >= attempt.deadline_at):
                try:
                    self.store.request_active_cancel(entry_id=entry.entry_id,
                                                    expected_revision=entry.revision,
                                                    actor=self.worker_id, now=self.clock())
                    entry = self.store.get_entry(entry.entry_id) or entry
                except Exception as exc:
                    return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                      f"runtime deadline passed but cancellation intent failed: {exc}")
            if entry.state == QueueState.CANCELING:
                result = self.executor.cancel(attempt.container_id or observation.container_id or "",
                                              run_id=attempt.run_id, attempt_id=attempt.attempt_id)
                if result.observation.state != "exited":
                    return WorkerTick("canceling", entry.entry_id, attempt.attempt_id,
                                      "termination not observed; queue slot remains held")
                return self._finish(entry, attempt, None, result.observation, canceled=True)
            return WorkerTick("active", entry.entry_id, attempt.attempt_id)
        if observation.state == "exited":
            # Persist terminal state and logs before attempting to load the cached
            # request. A missing verifier context must not erase observed facts.
            return self._finish(entry, attempt, None, observation)
        return WorkerTick("unknown", entry.entry_id, attempt.attempt_id,
                          f"unhandled observed state: {observation.state}")

    def _cancel_unstarted(self, entry: QueueEntry, attempt: QueueAttempt,
                          observation: ExecutionObservation) -> WorkerTick:
        try:
            log_ref = self.authority.persist_logs(entry, attempt,
                self.executor.logs(observation.container_id or attempt.container_id or ""))
            result = self.executor.cancel(observation.container_id or attempt.container_id or "",
                run_id=attempt.run_id, attempt_id=attempt.attempt_id)
            if result.observation.state != "absent":
                return WorkerTick("canceling", entry.entry_id, attempt.attempt_id,
                                  "unstarted container removal is unconfirmed; slot remains held")
            ref = self.authority.persist_worker_evidence(entry, attempt, "cancel-unstarted", {
                "command": list(result.command), "prior_state": "created", "observed_state": "absent",
                "exit_code": None, "log_ref": log_ref})
            updated = self._record(entry, attempt, ExternalOutcome.TERMINAL,
                                   log_ref=log_ref, evidence_ref=ref)
            failure = self.authority.record_failure(updated, attempt, None, result.observation)
            finalized = self.store.record_finalization(entry_id=entry.entry_id,
                expected_revision=updated.revision, outputs_verified=False,
                mlflow_finalized=failure.get("mlflow_finalized") is True,
                actor=self.worker_id, now=self.clock(),
                evidence_ref=str(failure.get("evidence_ref") or ref))
            return WorkerTick("canceled" if finalized.state == QueueState.CANCELED else "held",
                              entry.entry_id, attempt.attempt_id, finalized.blocking_reason)
        except Exception as exc:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              f"unstarted cancellation remains unresolved: {exc}")

    def _finish(self, entry: QueueEntry, attempt: QueueAttempt, spec: ContainerSpec | None,
                observation: ExecutionObservation, *, canceled: bool = False) -> WorkerTick:
        try:
            log_text = self.executor.logs(attempt.container_id or observation.container_id or "")
        except Exception as exc:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              f"terminal container observed but logs could not be preserved: {exc}")
        try:
            evidence_ref = self.authority.persist_logs(entry, attempt, log_text)
        except Exception as exc:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              f"terminal container observed but logs could not be durably preserved: {exc}")
        if not evidence_ref:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              "terminal container logs have no durable evidence reference")
        latest = self.store.get_entry(entry.entry_id) or entry
        current_attempt = self.store.get_attempt(attempt.attempt_id) or attempt
        outcome = ExternalOutcome.TERMINAL
        try:
            self._record(latest, current_attempt, outcome, exit_code=observation.exit_code,
                         log_ref=evidence_ref, evidence_ref=evidence_ref)
        except Exception as exc:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              f"could not persist terminal evidence: {exc}")
        latest = self.store.get_entry(entry.entry_id) or latest
        current_attempt = self.store.get_attempt(attempt.attempt_id) or current_attempt
        if canceled or latest.cancellation_requested:
            try:
                failure = self.authority.record_failure(latest, current_attempt, spec, observation)
            except Exception as exc:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  f"cancellation stopped the container but failure tracking did not finalize: {exc}")
            try:
                finalized = self.store.record_finalization(
                    entry_id=latest.entry_id, expected_revision=latest.revision,
                    outputs_verified=False, mlflow_finalized=failure.get("mlflow_finalized") is True,
                    actor=self.worker_id, now=self.clock(),
                    evidence_ref=str(failure.get("evidence_ref") or evidence_ref))
                return WorkerTick("canceled" if finalized.state == QueueState.CANCELED else "held",
                                  entry.entry_id, attempt.attempt_id, finalized.blocking_reason)
            except Exception as exc:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  f"termination observed; cancel transition failed: {exc}")
        if observation.exit_code != 0:
            try:
                failure = self.authority.record_failure(latest, current_attempt, spec, observation)
                failure_ref = str(failure.get("evidence_ref") or evidence_ref)
            except Exception as exc:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  f"container failed with code {observation.exit_code}; failure tracking failed: {exc}")
            try:
                failed = self.store.record_finalization(
                    entry_id=latest.entry_id, expected_revision=latest.revision,
                    outputs_verified=False,
                    mlflow_finalized=bool(failure.get("mlflow_finalized")),
                    actor=self.worker_id, now=self.clock(), evidence_ref=failure_ref)
                if failed.state == QueueState.FAILED:
                    return WorkerTick("failed", entry.entry_id, attempt.attempt_id,
                                      "confirmed trainer failure; configured continuation policy advanced FIFO")
                return WorkerTick("blocked", entry.entry_id, attempt.attempt_id,
                                  failed.blocking_reason or "confirmed trainer failure requires operator action")
            except Exception as exc:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  f"confirmed failure could not be committed: {exc}")
        if spec is None:
            try:
                spec = self.authority.read_prepared_execution(latest, current_attempt)
            except Exception as exc:
                return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                                  f"outputs cannot be verified because cached attempt context is unavailable: {exc}")
        try:
            verified = self.authority.verify_outputs(latest, current_attempt, spec)
        except Exception as exc:
            return self._block(latest, f"container exited zero but output/finalization verification failed: {exc}", evidence_ref)
        # Exit zero alone is never accepted; only the injected authoritative verifier
        # may return the evidence that allows queue completion.
        final_ref = str(verified.get("evidence_ref") or "")
        if not final_ref:
            return self._block(latest, "output verifier returned no durable evidence reference", evidence_ref)
        try:
            finalized = self.store.record_finalization(
                entry_id=latest.entry_id, expected_revision=latest.revision,
                outputs_verified=bool(verified.get("outputs_verified")),
                mlflow_finalized=bool(verified.get("mlflow_finalized")),
                actor=self.worker_id, now=self.clock(), evidence_ref=final_ref)
            if finalized.state == QueueState.COMPLETED:
                return self._cleanup_completed(finalized, current_attempt, spec)
            return WorkerTick("blocked", entry.entry_id, attempt.attempt_id,
                              finalized.blocking_reason or "output/finalization gate failed")
        except Exception as exc:
            return WorkerTick("held", entry.entry_id, attempt.attempt_id,
                              f"verified output evidence could not be committed: {exc}")

    def _cleanup_completed(self, entry: QueueEntry, attempt: QueueAttempt,
                           spec: ContainerSpec) -> WorkerTick:
        try:
            if attempt.container_id:
                result = self.executor.remove_verified_attempt(
                    attempt.run_id, attempt.attempt_id, attempt.container_id,
                    entry.intent.runtime.image_digest)
                if result.observation.state != "absent":
                    raise ExecutorError(result.detail or "exact owned container remains after cleanup",
                                        argv=result.command)
            self.authority.cleanup_verified_attempt(entry, attempt, spec)
            return WorkerTick("completed", entry.entry_id, attempt.attempt_id)
        except Exception as exc:
            try:
                ref = self.authority.persist_worker_evidence(entry, attempt, "cleanup-error", {
                    "command": list(getattr(exc, "argv", ())),
                    "config_sha256": entry.intent.config_sha256,
                    "image_digest": entry.intent.runtime.image_digest,
                    "container_name": attempt.container_name,
                    "error": str(exc), "detail": getattr(exc, "detail", ""),
                    "queue_state": QueueState.COMPLETED.value,
                })
            except Exception as persist_exc:
                ref = f"unavailable:{persist_exc}"
            return WorkerTick("completed", entry.entry_id, attempt.attempt_id,
                              f"training remains completed; bounded cleanup needs attention ({ref}): {exc}")

    def _block(self, entry: QueueEntry, reason: str, evidence_ref: str | None = None) -> WorkerTick:
        try:
            self.store.transition(entry_id=entry.entry_id, expected_revision=entry.revision,
                                  state=QueueState.BLOCKED, actor=self.worker_id,
                                  action="worker-hold", now=self.clock(),
                                  blocking_reason=reason, evidence_ref=evidence_ref)
        except Exception as exc:
            reason = f"{reason}; block record failed: {exc}"
        return WorkerTick("blocked", entry.entry_id, entry.active_attempt_id, reason)

    @staticmethod
    def _evidence(kind: str, value: str | None) -> str:
        """Return compact evidence text for stores without a configured log sink.

        Deployment integrations should replace this with a durable restricted log
        writer; never truncate engine command, digest, config, or container identity.
        """
        payload = json.dumps({"kind": kind, "value": value or ""}, sort_keys=True)
        return "inline:" + payload[:8192]
