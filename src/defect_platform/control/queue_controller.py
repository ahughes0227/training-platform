"""Integration of protected queue admission, persistence, and VM worker authority."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from defect_platform.catalog_store import DirectoryCatalogStore
from defect_platform.control.queue_admission import (
    QueueAdmission,
    QueueServiceConfig,
    load_queue_service_config,
    require_protected_file,
)
from defect_platform.control.queue_contracts import (
    QueueAttempt,
    QueueEntry,
    QueueIntent,
    QueuePolicyState,
    RetryClassification,
)
from defect_platform.control.queue_store import SQLiteQueueStore
from defect_platform.control.vm_executor import (
    BindMount,
    ContainerSpec,
    DiskCapacity,
    ExecutionObservation,
    HostIdentity,
    VMExecutor,
)
from defect_platform.semantics import canonical_bytes, make_training_manifest, read_manifest
from defect_platform.trainer.weights import verify_bundle_integrity


def _atomic_json(path: Path, value: dict) -> None:
    """Durable replace for private evidence, never follows a destination symlink."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o2770)
    if path.is_symlink():
        raise ValueError("evidence path cannot be a symlink")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(canonical_bytes(value))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp.unlink(missing_ok=True)


class AttemptTracker:
    """Persist intent before MLflow creation; an ambiguous creation is held.

    A recovered matching run can be reused. A search returning no run after a
    persisted intent cannot justify a second creation (search may be stale).
    """

    def __init__(self, config: QueueServiceConfig, *, client=None):
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from mlflow import MlflowClient
            self._client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
        return self._client

    def _auth(self):
        from defect_platform.mlflow_auth import mlflow_tracking_auth
        return mlflow_tracking_auth(self.config.mlflow_tracking_uri)

    def prepare(self, entry: QueueEntry, attempt: QueueAttempt, path: Path) -> str:
        tags = {"platform.run_id": entry.run_id, "platform.attempt_id": attempt.attempt_id,
                "platform.intent_sha256": entry.fingerprint,
                "dataset.sha256": entry.intent.dataset.sha256,
                "runtime.image_digest": entry.intent.runtime.image_digest}
        journal = json.loads(path.read_text()) if path.exists() else None
        with self._auth():
            experiment = self.client.get_experiment_by_name(self.config.mlflow_experiment_name)
            if experiment is None:
                raise ValueError("operator must provision the MLflow experiment before queue execution")
            matches = self.client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.`platform.attempt_id` = '{attempt.attempt_id}'",
                max_results=2)
            if len(matches) > 1:
                raise ValueError("multiple tracking runs match attempt; explicit reconciliation required")
            if matches:
                run = matches[0]
                if any(run.data.tags.get(key) != value for key, value in tags.items()):
                    raise ValueError("recovered tracking run has mismatched immutable lineage")
                if journal and journal.get("run_id") not in {None, run.info.run_id}:
                    raise ValueError("tracking journal differs from recovered attempt")
            elif journal:
                raise ValueError("MLflow create outcome remains unknown; refusing duplicate creation")
            else:
                _atomic_json(path, {"state": "create_intent", "tags": tags})
                run = self.client.create_run(experiment.experiment_id, tags=tags,
                                             run_name=entry.intent.experiment.experiment_id)
            _atomic_json(path, {"state": "created", "run_id": run.info.run_id, "tags": tags})
            return run.info.run_id

    def verify(self, run_id: str, attempt: QueueAttempt, fingerprint: str) -> None:
        with self._auth():
            run = self.client.get_run(run_id)
            if (run.info.status != "FINISHED"
                    or run.data.tags.get("platform.attempt_id") != attempt.attempt_id
                    or run.data.tags.get("platform.intent_sha256") != fingerprint):
                raise ValueError("MLflow run is not finalized with exact attempt lineage")

    def fail(self, run_id: str, *, canceled: bool = False) -> None:
        with self._auth():
            self.client.set_terminated(run_id, status="KILLED" if canceled else "FAILED")


class GCSOutputVerifier:
    def __init__(self, client=None):
        self.client = client

    def verify(self, directory: Path, destination: str) -> None:
        if self.client is None:
            from google.cloud import storage
            self.client = storage.Client()
        bucket, _, prefix = destination[5:].partition("/")
        expected = {}
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("training output contains a symlink")
            if path.is_file():
                name = prefix.rstrip("/") + "/" + path.relative_to(directory).as_posix()
                expected[name] = path
        observed = {blob.name: blob for blob in self.client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/")}
        if set(observed) != set(expected):
            raise ValueError("remote output set differs from complete local attempt output")
        for name, path in expected.items():
            remote = hashlib.sha256()
            with observed[name].open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    remote.update(chunk)
            local = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    local.update(chunk)
            if remote.digest() != local.digest():
                raise ValueError(f"remote artifact integrity mismatch: {name}")


class QueueController:
    def __init__(self, *, config: QueueServiceConfig, store: SQLiteQueueStore,
                 admission: QueueAdmission, tracker=None, artifact_verifier=None,
                 clock=lambda: datetime.now(UTC), capacity_provider=None, backup_client=None):
        self.config, self.store, self.admission = config, store, admission
        self.storage_path, self.clock = config.storage_path, clock
        self.tracker = tracker or AttemptTracker(config)
        self.artifact_verifier = artifact_verifier or GCSOutputVerifier()
        self.capacity_provider = capacity_provider or self._capacity
        self.backup_client = backup_client
        store.initialize(scope=config.scope, operator_config=config.operator,
                         actor=config.worker_id, now=clock())

    def _capacity(self) -> DiskCapacity:
        value = os.statvfs(self.storage_path)
        return DiskCapacity(self.storage_path, value.f_bavail * value.f_frsize,
                            value.f_favail, value.f_blocks * value.f_frsize)

    def _evidence(self, kind: str, value: dict) -> str:
        path = self.storage_path / "evidence" / f"{kind}-{uuid.uuid4().hex}.json"
        _atomic_json(path, value)
        return str(path)

    def record_capacity(self, capacity: DiskCapacity, observed_at: datetime) -> str:
        return self._evidence("capacity", {"at": observed_at.isoformat(),
            "path": str(capacity.path), "free_bytes": capacity.free_bytes,
            "free_inodes": capacity.free_inodes, "capacity_bytes": capacity.capacity_bytes})

    def validate_executor(self, executor: VMExecutor, identity: HostIdentity) -> None:
        readiness = self.admission.refresh().readiness if self.admission.refresh else self.config.readiness
        if identity != HostIdentity(readiness.project_id, readiness.zone,
                                    readiness.instance_id, readiness.boot_id):
            raise ValueError("current host differs from approved runtime readiness")
        executor.assert_readiness(readiness, gpu_devices=tuple(self.config.gpu_devices))

    def heartbeat(self, result) -> None:
        _atomic_json(self.storage_path / "worker-status.json", {
            "at": self.clock().isoformat(), "action": result.action,
            "entry_id": result.entry_id, "attempt_id": result.attempt_id, "reason": result.reason})

    def record_exposure(self, now: datetime) -> None:
        policy = self.store.policy(self.config.scope)
        ref = self._evidence("cost", {"until": now.isoformat(),
            "authorization": self.config.operator.campaign_authorization_id})
        self.store.account_until(scope=self.config.scope,
            expected_policy_revision=policy["revision"], observed_at=now,
            actor=self.config.worker_id, evidence_ref=ref)

    def enqueue(self, intent: QueueIntent, *, idempotency_key: str, requester: str) -> QueueEntry:
        if idempotency_key != intent.idempotency_key:
            raise ValueError("prepared idempotency key differs from request")
        return self.enqueue_batch([intent], idempotency_key="single:" + idempotency_key,
                                  requester=requester)["entries"][0]

    def enqueue_batch(self, intents: list[QueueIntent], *, idempotency_key: str,
                      requester: str) -> dict:
        if not idempotency_key or not requester:
            raise ValueError("batch key and authenticated requester are required")
        now = self.clock()
        for intent in intents:
            self.admission.validate(intent, now=now)
        self.record_exposure(now)
        policy = self.store.policy(self.config.scope)
        capacity = self.capacity_provider()
        capacity_ref = self.record_capacity(capacity, now)
        prior_ids = {e.entry_id for e in self.store.list_entries(self.config.scope)}
        entries = self.store.enqueue_batch(scope=self.config.scope, intents=intents,
            batch_idempotency_key=idempotency_key,
            operator_config=self.config.operator, expected_policy_revision=policy["revision"],
            actor=requester, now=now, observed_free_bytes=capacity.free_bytes,
            observed_free_inodes=capacity.free_inodes, capacity_evidence_ref=capacity_ref,
            capacity_observed_at=now)
        return {"entries": entries, "created": [e.entry_id not in prior_ids for e in entries]}

    def list_queue(self) -> dict:
        entries = self.store.list_entries(self.config.scope)
        return {"scope": self.config.scope, "entries": [e.model_dump(mode="json") for e in entries],
                "attempts": [a.model_dump(mode="json", exclude={"claim_token"}) for e in entries
                             for a in self.store.list_attempts(e.entry_id)], **self.status()}

    def status(self) -> dict:
        policy = self.store.policy(self.config.scope)
        now = self.clock()
        entries = self.store.list_entries(self.config.scope)
        active = next((e for e in entries if e.state.value in {
            "dispatching", "active", "canceling", "blocked"}), None)
        attempt = self.store.get_attempt(active.active_attempt_id) if active and active.active_attempt_id else None
        reserved = sum(e.budget_reserved_usd for e in entries if e.state.value not in {"completed", "canceled", "failed"})
        observations = [e.last_observation_at for e in entries if e.last_observation_at]
        heartbeat_path = self.storage_path / "worker-status.json"
        heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else None
        head = next((e for e in entries if e.state.value not in {"completed", "canceled", "failed"}), None)
        previous_finish = max((e.updated_at for e in entries if head and e.sequence < head.sequence), default=None)
        eligible_at = max([head.created_at, *(stamp for stamp in (head.next_eligible_at, previous_finish) if stamp)]) if head else None
        return {**policy, "operator_config": policy["operator_config"].model_dump(mode="json"),
            "current_vm": {"project": self.config.readiness.project_id,
            "zone": self.config.readiness.zone, "instance_id": self.config.readiness.instance_id,
            "boot_id": self.config.readiness.boot_id},
            "readiness_expires_at": self.config.readiness.expires_at.isoformat(),
            "readiness_current": self.config.readiness.observed_at <= now < self.config.readiness.expires_at,
            "current_container": attempt.container_id if attempt else None,
            "current_run_id": active.run_id if active else None,
            "blocking_reason": active.blocking_reason if active else None,
            "budget_reserved_usd": reserved,
            "budget_remaining_usd": self.config.operator.campaign_cost_cap_usd - policy["spent_usd"] - reserved,
            "last_observation": max(observations).isoformat() if observations else None,
            "observation_stale": not observations or (now - max(observations)).total_seconds() > self.config.operator.stalled_work_seconds,
            "worker": heartbeat,
            "head_entry_id": head.entry_id if head else None,
            "head_eligible_at": eligible_at.isoformat() if eligible_at else None,
            "start_delay_exceeded": bool(head and head.state.value == "waiting"
                and active is None and policy["state"] == "running" and eligible_at
                and (now - eligible_at).total_seconds() > self.config.operator.max_start_delay_seconds),
            "worker_stale": heartbeat is None or (now - datetime.fromisoformat(heartbeat["at"])).total_seconds() > self.config.operator.stalled_work_seconds,
            "backup": json.loads((self.storage_path / "backup-status.json").read_text()) if (self.storage_path / "backup-status.json").exists() else None,
            "observed_at": now.isoformat(),
            "required_action": "Inspect blocked entries and live host evidence before recovery"}

    def _entry(self, entry_id: str) -> QueueEntry:
        entry = self.store.get_entry(entry_id)
        if entry is None or entry.scope != self.config.scope:
            raise KeyError("entry is outside this queue scope")
        return entry

    def pause(self, *, reason: str, actor: str) -> dict:
        ref = self._evidence("pause", {"reason": reason, "actor": actor})
        policy = self.store.policy(self.config.scope)
        self.store.set_policy(scope=self.config.scope, expected_policy_revision=policy["revision"],
                              state=QueuePolicyState.PAUSED, actor=actor, now=self.clock())
        return {**self.status(), "evidence_ref": ref}

    def resume(self, *, actor: str) -> dict:
        for entry in self.store.list_entries(self.config.scope):
            if entry.state.value == "waiting":
                self.admission.validate(entry.intent, now=self.clock())
                break
        self.record_exposure(self.clock())
        policy = self.store.policy(self.config.scope)
        if policy["spent_usd"] >= self.config.operator.campaign_cost_cap_usd:
            raise ValueError("campaign spending authority is exhausted")
        self.store.set_policy(scope=self.config.scope, expected_policy_revision=policy["revision"],
                              state=QueuePolicyState.RUNNING, actor=actor, now=self.clock())
        return self.status()

    def cancel_waiting(self, entry_id: str, *, actor: str) -> QueueEntry:
        entry = self._entry(entry_id)
        return self.store.cancel_waiting(entry_id=entry_id, expected_revision=entry.revision,
                                        actor=actor, now=self.clock())

    def request_active_cancel(self, entry_id: str, *, actor: str) -> QueueEntry:
        entry = self._entry(entry_id)
        return self.store.request_active_cancel(entry_id=entry_id, expected_revision=entry.revision,
                                                actor=actor, now=self.clock())

    def retry(self, entry_id: str, *, expected_revision: int, reason: str, actor: str,
              classification: RetryClassification = RetryClassification.UNKNOWN) -> QueueEntry:
        entry = self._entry(entry_id)
        if not reason.strip():
            raise ValueError("retry classification requires an operator reason")
        classification = RetryClassification(classification)
        self.admission.validate(entry.intent, now=self.clock())
        return self.store.retry(entry_id=entry_id, expected_revision=expected_revision, actor=actor,
            now=self.clock(), classification=classification, evidence_ref=self._evidence("retry", {
                "actor": actor, "entry": entry_id, "reason": reason,
                "classification": classification.value, "attempt": entry.active_attempt_id}))

    def skip(self, entry_id: str, *, expected_revision: int, reason: str, actor: str) -> QueueEntry:
        self._entry(entry_id)
        return self.store.skip(entry_id=entry_id, expected_revision=expected_revision, actor=actor,
            now=self.clock(), evidence_ref=self._evidence("skip", {"actor": actor, "reason": reason}))

    def record_observation(self, observation: dict, *, worker_id: str) -> QueueEntry:
        # Engine observations must carry the exact fenced persisted attempt capability.
        allowed = {"entry_id", "expected_revision", "attempt_id", "dispatch_id", "claim_token",
                   "external_outcome", "container_id", "process_id", "exit_code", "log_ref",
                   "output_ref", "evidence_ref"}
        if set(observation) - allowed:
            raise ValueError("unrecognized worker observation fields")
        self._entry(observation["entry_id"])
        attempt = self.store.get_attempt(observation["attempt_id"])
        if (attempt is None or attempt.worker_id != self.config.worker_id
                or worker_id not in {f"worker-uid:{uid}" for uid in self.config.worker_uids}):
            raise PermissionError("observation principal differs from claimed worker")
        from defect_platform.control.queue_contracts import ExternalOutcome
        observation = {**observation, "external_outcome": ExternalOutcome(observation["external_outcome"])}
        return self.store.record_dispatch(**observation, actor=attempt.worker_id, now=self.clock())

    def resolve_interrupted(self, entry_id: str, *, expected_revision: int,
                            reason: str, actor: str, executor=None) -> QueueEntry:
        """Explicit root-administrator recovery; never dispatched by the API.

        Absence alone is insufficient until instance identity, engine state and
        the host GPU process pool have been observed under the worker lock.
        """
        if os.geteuid() != 0 or actor != "admin-uid:0":
            raise PermissionError("interrupted recovery requires the Linux root administrator")
        from defect_platform.control.vm_executor import compute_engine_identity
        fd = os.open(self.storage_path / "worker.lock", os.O_CREAT | os.O_RDWR, 0o660)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            entry = self._entry(entry_id)
            if entry.revision != expected_revision or not entry.active_attempt_id:
                raise ValueError("recovery revision changed or no attempt is recorded")
            attempt = self.store.get_attempt(entry.active_attempt_id)
            if attempt is None:
                raise ValueError("attempt identity is missing")
            if executor is None:
                current = compute_engine_identity()
                executor = VMExecutor(expected_identity=current, identity_provider=compute_engine_identity)
            current = executor.assert_host_identity()
            if (current.project, current.zone, current.instance_id) != (
                    attempt.project_id, attempt.zone, attempt.instance_id):
                raise ValueError("instance replacement requires a separate recovery handoff")
            observed = executor.inspect(entry.run_id, attempt.attempt_id, attempt.container_id)
            if observed.state not in {"absent", "exited"}:
                raise ValueError("previous container remains active or unobservable")
            if executor.scan_gpu_workloads():
                raise ValueError("GPU workload pool is not proven empty; recovery held")
            ref = self._evidence("administrator-recovery", {"actor": actor, "reason": reason,
                "entry": entry_id, "attempt": attempt.attempt_id, "previous_boot": attempt.boot_id,
                "current_boot": current.boot_id, "container_state": observed.state,
                "observed_exit_code": observed.exit_code, "container": attempt.container_id})
            return self.store.mark_interrupted(entry_id=entry_id, expected_revision=expected_revision,
                actor=actor, now=self.clock(), evidence_ref=ref)
        finally:
            os.close(fd)

    def _attempt_paths(self, entry: QueueEntry, attempt: QueueAttempt) -> tuple[Path, Path, Path]:
        for value in (entry.run_id, attempt.attempt_id):
            if str(uuid.UUID(value)) != value:
                raise ValueError("invalid persisted attempt path identity")
        base = self.storage_path / "attempts" / entry.run_id / attempt.attempt_id
        paths = (base / "input", base / "output", base / "scratch")
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(self.storage_path.resolve()):
                raise ValueError("attempt directory escaped protected storage")
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.geteuid() == 0:
                os.chown(path, self.config.trainer_uid, self.config.trainer_gid)
        return paths

    def prepare_execution(self, entry: QueueEntry, attempt: QueueAttempt,
                          identity: HostIdentity) -> ContainerSpec:
        semantic = self.admission.validate(entry.intent, now=self.clock(), identity=identity)
        input_dir, output_dir, scratch_dir = self._attempt_paths(entry, attempt)
        request_path = input_dir / "request.json"
        output_uri = f"{self.config.output_root_uri.rstrip('/')}/{entry.run_id}/{attempt.attempt_id}"
        payload = {"experiment": entry.intent.experiment.model_dump(mode="json"),
                   "dataset": entry.intent.dataset.model_dump(mode="json"),
                   "runtime": entry.intent.runtime.model_dump(mode="json"),
                   "classes": entry.intent.classes,
                   "semantic_refs": {"dataset_semantic_sha256": semantic.sha256,
                                     "catalog_sha256": semantic.catalog.sha256},
                   "output_uri": output_uri, "mlflow_tracking_uri": self.config.mlflow_tracking_uri}
        # Tracking/evidence are outside trainer-writable paths.
        journal = self.storage_path / "tracking" / f"{attempt.attempt_id}.json"
        payload["mlflow_run_id"] = self.tracker.prepare(entry, attempt, journal)
        if request_path.exists():
            if request_path.is_symlink() or request_path.read_bytes() != canonical_bytes(payload):
                raise ValueError("immutable attempt request differs from prepared payload")
        else:
            _atomic_json(request_path, payload)
            if os.geteuid() == 0:
                os.chown(request_path, self.config.trainer_uid, self.config.trainer_gid)
            request_path.chmod(0o400)
        require_protected_file(self.config.credential_path)
        return self._container_spec(entry, attempt, input_dir, output_dir, scratch_dir)

    def read_prepared_execution(self, entry: QueueEntry, attempt: QueueAttempt) -> ContainerSpec:
        base = self.storage_path / "attempts" / entry.run_id / attempt.attempt_id
        input_dir, output_dir, scratch_dir = base / "input", base / "output", base / "scratch"
        if not base.resolve().is_relative_to(self.storage_path.resolve()):
            raise ValueError("attempt path escaped protected storage")
        request_path = input_dir / "request.json"
        if request_path.is_symlink():
            raise ValueError("immutable prepared request is a symlink")
        payload = json.loads(request_path.read_text())
        for key in ("experiment", "dataset", "runtime", "classes"):
            expected = getattr(entry.intent, key)
            expected = expected.model_dump(mode="json") if hasattr(expected, "model_dump") else expected
            if payload.get(key) != expected:
                raise ValueError("immutable prepared request differs from persisted queue intent")
        if payload.get("output_uri") != f"{self.config.output_root_uri.rstrip('/')}/{entry.run_id}/{attempt.attempt_id}":
            raise ValueError("prepared output path differs from exact protected attempt location")
        return self._container_spec(entry, attempt, input_dir, output_dir, scratch_dir)

    def _container_spec(self, entry: QueueEntry, attempt: QueueAttempt,
                        input_dir: Path, output_dir: Path, scratch_dir: Path) -> ContainerSpec:
        request_path = input_dir / "request.json"
        return ContainerSpec(run_id=entry.run_id, attempt_id=attempt.attempt_id,
            image_digest=entry.intent.runtime.image_digest,
            command=("python", "-m", "defect_platform.trainer.runner", "--request",
                     "/defect/input/request.json", "--output-dir", "/defect/output"),
            output_dir=output_dir, scratch_dir=scratch_dir,
            inputs=(BindMount(request_path, "/defect/input/request.json", True),),
            credential_mounts=(BindMount(self.config.credential_path, "/defect/credentials.json", True),),
            environment=(("GOOGLE_APPLICATION_CREDENTIALS", "/defect/credentials.json"),
                         ("HF_HOME", "/defect/scratch/huggingface"),
                         ("TORCH_HOME", "/defect/scratch/torch")),
            uid_gid=f"{self.config.trainer_uid}:{self.config.trainer_gid}",
            gpu_devices=tuple(self.config.gpu_devices), network_mode=self.config.network_name,
            memory_bytes=self.config.memory_bytes, pids_limit=self.config.pids_limit,
            timeout_seconds=entry.intent.job.max_runtime_seconds,
            metadata_isolation_verified=bool(self.config.readiness.metadata_denial_evidence_ref))

    def cleanup_verified_attempt(self, entry: QueueEntry, attempt: QueueAttempt,
                                 spec: ContainerSpec) -> str:
        current = self._entry(entry.entry_id)
        if (current.state.value != "completed" or current.container_exit_code != 0
                or not current.outputs_verified or not current.mlflow_finalized
                or current.active_attempt_id != attempt.attempt_id):
            raise ValueError("cleanup requires persisted verified success for this exact attempt")
        path = spec.scratch_dir
        expected = self.storage_path / "attempts" / entry.run_id / attempt.attempt_id / "scratch"
        if path != expected or path.is_symlink() or not path.resolve().is_relative_to(self.storage_path.resolve()):
            raise ValueError("scratch cleanup path is outside the verified attempt")
        if path.exists():
            shutil.rmtree(path)
        return self._evidence("scratch-cleaned", {"entry": entry.entry_id, "attempt": attempt.attempt_id,
                                                 "path": str(path), "local_outputs_retained": True})

    def persist_logs(self, entry: QueueEntry, attempt: QueueAttempt, text: str) -> str:
        return self._evidence("container-logs", {"entry": entry.entry_id,
            "attempt": attempt.attempt_id, "digest": entry.intent.runtime.image_digest,
            "container": attempt.container_id, "logs": text})

    def persist_worker_evidence(self, entry: QueueEntry, attempt: QueueAttempt,
                                kind: str, payload: dict) -> str:
        return self._evidence("worker-" + kind, {
            "entry_id": entry.entry_id, "attempt_id": attempt.attempt_id,
            "runtime_digest": entry.intent.runtime.image_digest,
            "configuration_id": entry.fingerprint, "container_name": attempt.container_name,
            "claim_token_sha256": hashlib.sha256(attempt.claim_token.encode()).hexdigest(),
            "at": self.clock().isoformat(), **payload})

    def verify_outputs(self, entry: QueueEntry, attempt: QueueAttempt, spec: ContainerSpec) -> dict:
        semantic = self.admission.validate(entry.intent, now=self.clock())
        report_path = spec.output_dir / "evaluation.json"
        if report_path.is_symlink():
            raise ValueError("evaluation output cannot be a symlink")
        report = json.loads(report_path.read_text(), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
        self._validate_report(report, entry)
        expected = make_training_manifest(semantic, entry.intent.experiment,
            dataset_sha256=entry.intent.dataset.sha256,
            runtime_image_digest=entry.intent.runtime.image_digest,
            runtime_source_commit=entry.intent.runtime.source_commit)
        model_manifest = read_manifest(spec.output_dir / "model" / "semantics.json")
        if model_manifest.sha256 != expected.sha256 or report.get("semantic_sha256") != expected.sha256:
            raise ValueError("model semantics differ from exact dataset/configuration/runtime intent")
        if report.get("classes") != entry.intent.classes or report.get("device") != "cuda" or report.get("visible_gpu_count", 0) < 1:
            raise ValueError("training report lacks expected class order or actual GPU execution")
        mapping = json.loads((spec.output_dir / "model" / "class_mapping.json").read_text())
        if mapping != {"classes": entry.intent.classes}:
            raise ValueError("portable model class mapping differs from canonical order")
        verify_bundle_integrity(spec.output_dir / "model")
        for name in ("model.pt", "inference_config.json", "backbone/config.json"):
            if not (spec.output_dir / "model" / name).is_file():
                raise ValueError(f"portable model bundle is incomplete: {name}")
        if not any((spec.output_dir / "model" / "backbone").glob("*.safetensors")) and not any((spec.output_dir / "model" / "backbone").glob("*.bin")):
            raise ValueError("portable backbone weights are missing")
        request = json.loads((spec.inputs[0].source).read_text())
        if report.get("mlflow_run_id") != request["mlflow_run_id"] or report.get("output_uri") != request["output_uri"]:
            raise ValueError("reported outputs/tracking do not match this attempt")
        self.artifact_verifier.verify(spec.output_dir, request["output_uri"])
        self.tracker.verify(request["mlflow_run_id"], attempt, entry.fingerprint)
        ref = self._evidence("completion", {"entry": entry.entry_id, "attempt": attempt.attempt_id,
            "semantic_sha256": expected.sha256, "output_uri": request["output_uri"],
            "mlflow_run_id": request["mlflow_run_id"], "release_eligible": False})
        return {"evidence_ref": ref, "outputs_verified": True, "mlflow_finalized": True}

    @staticmethod
    def _validate_report(report: dict, entry: QueueEntry) -> None:
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("evaluation contains a nonfinite number")
            if isinstance(value, dict):
                for child in value.values():
                    finite(child)
            elif isinstance(value, list):
                for child in value:
                    finite(child)
        finite(report)
        if not isinstance(report.get("best_epoch"), int) or not 1 <= report["best_epoch"] <= entry.intent.experiment.epochs:
            raise ValueError("evaluation best epoch is outside the submitted experiment")
        if not report.get("history"):
            raise ValueError("evaluation has no training history")
        count = len(entry.intent.classes)
        for split in ("validation", "test"):
            metrics = report.get(split, {})
            for metric, minimum in (("mcc", -1), ("macro_f1", 0), ("accuracy", 0)):
                value = metrics.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= 1:
                    raise ValueError(f"evaluation {split}/{metric} is missing or invalid")
            confusion = metrics.get("confusion")
            if (not isinstance(confusion, list) or len(confusion) != count
                    or any(not isinstance(row, list) or len(row) != count for row in confusion)
                    or any(type(value) is not int or value < 0 for row in confusion for value in row)
                    or sum(map(sum, confusion)) == 0
                    or set(metrics.get("per_class", {})) != set(entry.intent.classes)):
                raise ValueError("evaluation confusion/per-class report is incoherent")
        from defect_platform.semantics import policy_for_config
        fields = ("optimizer", "loss", "focal_gamma", "horizontal_flip_probability",
                  "class_weights", "seed", "max_review_error_rate")
        policy = policy_for_config(entry.intent.experiment).model_dump(mode="json")
        if report.get("training_config") != {key: policy[key] for key in fields}:
            raise ValueError("reported training policy differs from submitted experiment")

    def record_failure(self, entry: QueueEntry, attempt: QueueAttempt,
                       spec: ContainerSpec | None, observation: ExecutionObservation) -> dict:
        journal = self.storage_path / "tracking" / f"{attempt.attempt_id}.json"
        if not journal.exists():
            raise ValueError("failed attempt has no recoverable tracking identity")
        tracking_run_id = json.loads(journal.read_text())["run_id"]
        self.tracker.fail(tracking_run_id, canceled=entry.cancellation_requested)
        ref = self._evidence("failure", {"entry": entry.entry_id, "attempt": attempt.attempt_id,
            "exit_code": observation.exit_code, "runtime": entry.intent.runtime.image_digest,
            "config_sha256": entry.fingerprint, "log_ref": attempt.log_ref,
            "tracking_run_id": tracking_run_id, "mlflow_finalized": True,
            "outputs_verified": False, "cancellation_requested": entry.cancellation_requested,
            "observed_at": self.clock().isoformat()})
        return {"evidence_ref": ref, "outputs_verified": False, "mlflow_finalized": True}

    def backup(self, destination: Path) -> dict:
        root = self.storage_path / "backups"
        if not destination.is_absolute() or not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("backup destination must be under protected queue backups directory")
        root.mkdir(parents=True, exist_ok=True, mode=0o2770)
        return {"backup": self.store.backup(str(destination)), "off_vm_copy_required": True}

    def verify_restore(self, source: Path) -> dict:
        root = self.storage_path / "backups"
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("restore check must use the protected backups directory")
        return self.store.verify_restore(str(source)).model_dump(
            mode="json", exclude={"attempts": {"__all__": {"claim_token"}}})

    def maintain_backup(self, now: datetime) -> str:
        """Off-VM recovery archive, excluding credentials and trainer scratch/models.

        Upload uses a create-only object name. Publication is recorded only after
        its remote content hash matches. Live bucket retention/deletion policy
        remains an infrastructure-owner acceptance gate.
        """
        journal = self.storage_path / "backup-status.json"
        previous = json.loads(journal.read_text()) if journal.exists() else None
        if previous and (now - datetime.fromisoformat(previous["at"])).total_seconds() < self.config.backup_interval_seconds:
            return previous["uri"]
        root = self.storage_path / "backups"
        root.mkdir(parents=True, exist_ok=True, mode=0o2770)
        archive = root / f"{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory(dir=root) as temp:
            snapshot = Path(temp) / "queue.sqlite"
            self.store.backup(str(snapshot))
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(snapshot, arcname="queue.sqlite")
                for directory in (self.storage_path / "evidence", self.storage_path / "tracking"):
                    if directory.exists():
                        for path in directory.rglob("*"):
                            if path.is_symlink():
                                raise ValueError("recovery evidence contains a symlink")
                            if path.is_file():
                                bundle.add(path, arcname=path.relative_to(self.storage_path).as_posix())
                for path in (self.storage_path / "attempts").glob("*/*/input/request.json"):
                    if path.is_symlink():
                        raise ValueError("immutable request is a symlink")
                    bundle.add(path, arcname=path.relative_to(self.storage_path).as_posix())
                config = Path(temp) / "operator-config.json"
                config.write_bytes(canonical_bytes(self.config.model_dump(mode="json")))
                bundle.add(config, arcname="operator-config.json")
        archive.chmod(0o640)
        if self.backup_client is None:
            from google.cloud import storage
            self.backup_client = storage.Client()
        bucket, _, prefix = self.config.backup_root_uri[5:].partition("/")
        key = prefix.rstrip("/") + "/" + archive.name
        blob = self.backup_client.bucket(bucket).blob(key)
        blob.upload_from_filename(str(archive), if_generation_match=0)
        local = hashlib.sha256()
        with archive.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                local.update(chunk)
        digest = local.hexdigest()
        remote = hashlib.sha256()
        with blob.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                remote.update(chunk)
        if remote.hexdigest() != digest:
            raise ValueError("off-VM recovery archive integrity mismatch")
        uri = f"gs://{bucket}/{key}"
        _atomic_json(journal, {"at": now.isoformat(), "uri": uri, "sha256": digest})
        # Remove only verified-upload archives; preserve unuploaded fault evidence.
        uploaded = root / "uploaded"
        uploaded.mkdir(exist_ok=True, mode=0o2750)
        os.replace(archive, uploaded / archive.name)
        old = sorted(uploaded.glob("*.tar.gz"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in old[self.config.backup_local_keep:]:
            path.unlink()
        return uri


def create_vm_queue_controller(config_path: Path) -> QueueController:
    config = load_queue_service_config(config_path)
    os.umask(0o007)
    # Deployment provisions this directory. Only trusted control services share
    # the state group; operators and trainers must not be group members.
    info = config.storage_path.lstat()
    if (config.storage_path.is_symlink() or not config.storage_path.is_dir()
            or info.st_mode & 0o007 or info.st_gid != config.state_group_id
            or info.st_uid not in {0, os.geteuid()} or not info.st_mode & 0o2000):
        raise PermissionError("queue state requires protected preprovisioned setgid directory")
    for parent in config.storage_path.parents:
        parent_info = parent.lstat()
        if parent.is_symlink() or parent_info.st_mode & 0o022:
            raise PermissionError("queue state parent is writable or a symlink")
    require_protected_file(config_path)
    store = SQLiteQueueStore(config.storage_path / "queue.sqlite")
    catalogs = DirectoryCatalogStore(config.catalog_root)
    admission = QueueAdmission(config, catalogs,
                               refresh=lambda: load_queue_service_config(config_path))
    return QueueController(config=config, store=store, admission=admission)


def create_vm_queue_worker(config_path: Path):
    from defect_platform.control.queue_worker import QueueWorker
    from defect_platform.control.vm_executor import compute_engine_identity
    controller = create_vm_queue_controller(config_path)
    config = controller.config
    executor = VMExecutor(expected_identity=HostIdentity(
        config.readiness.project_id, config.readiness.zone, config.readiness.instance_id,
        config.readiness.boot_id), identity_provider=compute_engine_identity)
    return QueueWorker(store=controller.store, executor=executor, authority=controller,
        scope=config.scope, worker_id=config.worker_id, lock_path=config.storage_path / "worker.lock",
        lease_seconds=config.lease_seconds)
