"""Single-authority transactional SQLite persistence for the VM training queue."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from defect_platform.control.queue_contracts import (
    AttemptState,
    ExternalOutcome,
    QueueAttempt,
    QueueClaim,
    QueueEntry,
    QueueEvent,
    QueueIntent,
    QueueOperatorConfig,
    QueuePolicyState,
    QueueSnapshot,
    QueueState,
    RetryClassification,
)


class QueueConflict(RuntimeError):
    """A compare-and-swap or pinned-policy precondition no longer holds."""


class QueueAdmissionError(ValueError):
    """A configured capacity, budget, or backlog gate denies admission."""


class SQLiteQueueStore:
    """Durable single-VM FIFO queue; unknown external outcomes remain held."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = str(path)
        db_path = Path(self.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists() and db_path.stat().st_size:
            # Inspect any existing database read-only before CREATE TABLE can
            # mutate it. Unknown layouts and versions are fail-closed.
            readonly_uri = f"file:{db_path.resolve()}?mode=ro"
            readonly = sqlite3.connect(readonly_uri, uri=True)
            try:
                tables = {row[0] for row in readonly.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
                if "queue_scopes" in tables:
                    columns = {row[1] for row in readonly.execute("PRAGMA table_info(queue_scopes)")}
                    if "schema_version" not in columns:
                        raise ValueError("unsupported queue database schema: missing schema_version")
                    versions = {row[0] for row in readonly.execute(
                        "SELECT DISTINCT schema_version FROM queue_scopes")}
                    if any(version != self.SCHEMA_VERSION for version in versions):
                        raise ValueError(f"unsupported queue schema version(s): {sorted(versions)}")
                elif tables:
                    raise ValueError("database contains tables but no recognized queue schema")
            finally:
                readonly.close()
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS queue_scopes (
                    scope TEXT PRIMARY KEY, policy_state TEXT NOT NULL,
                    policy_revision INTEGER NOT NULL, next_sequence INTEGER NOT NULL,
                    operator_config TEXT NOT NULL, schema_version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS queue_entries (
                    entry_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                    sequence INTEGER NOT NULL, idempotency_key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, record TEXT NOT NULL,
                    UNIQUE(scope, sequence), UNIQUE(scope, idempotency_key));
                CREATE INDEX IF NOT EXISTS queue_entries_order
                    ON queue_entries(scope, sequence);
                CREATE TABLE IF NOT EXISTS queue_batches (
                    scope TEXT NOT NULL, batch_key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, entry_ids TEXT NOT NULL,
                    PRIMARY KEY(scope, batch_key));
                CREATE TABLE IF NOT EXISTS queue_attempts (
                    attempt_id TEXT PRIMARY KEY, entry_id TEXT NOT NULL,
                    record TEXT NOT NULL,
                    FOREIGN KEY(entry_id) REFERENCES queue_entries(entry_id));
                CREATE INDEX IF NOT EXISTS queue_attempts_entry
                    ON queue_attempts(entry_id);
                CREATE TABLE IF NOT EXISTS queue_events (
                    ordinal INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL,
                    entry_id TEXT, record TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS queue_exposure (
                    scope TEXT PRIMARY KEY, spent_usd REAL NOT NULL, revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, last_evidence_ref TEXT NOT NULL);
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _fingerprint(intent: QueueIntent) -> str:
        value = intent.model_dump(mode="json", exclude={"idempotency_key"})
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _event(*, scope: str, actor: str, now: datetime, action: str,
               resulting_revision: int, entry_id: str | None = None,
               attempt_id: str | None = None, prior_revision: int | None = None,
               configuration_id: str | None = None, runtime_digest: str | None = None,
               vm_identity: str | None = None, container_identity: str | None = None,
               process_identity: str | None = None, log_ref: str | None = None,
               evidence_ref: str | None = None, detail: str | None = None) -> QueueEvent:
        return QueueEvent(event_id=str(uuid.uuid4()), scope=scope, entry_id=entry_id,
                          attempt_id=attempt_id, actor=actor, at=now, action=action,
                          prior_revision=prior_revision, resulting_revision=resulting_revision,
                          configuration_id=configuration_id, runtime_digest=runtime_digest,
                          vm_identity=vm_identity, container_identity=container_identity,
                          process_identity=process_identity, log_ref=log_ref,
                          evidence_ref=evidence_ref, detail=detail)

    @staticmethod
    def _append_event(db: sqlite3.Connection, event: QueueEvent) -> None:
        db.execute("INSERT INTO queue_events(scope,entry_id,record) VALUES(?,?,?)",
                   (event.scope, event.entry_id, SQLiteQueueStore._json(event)))

    @staticmethod
    def _scope_row(db: sqlite3.Connection, scope: str):
        row = db.execute("SELECT * FROM queue_scopes WHERE scope=?", (scope,)).fetchone()
        if row is None:
            raise KeyError(f"queue scope is not configured: {scope}")
        return row

    def initialize(self, *, scope: str, operator_config: QueueOperatorConfig,
                   actor: str, now: datetime) -> None:
        if not scope or not actor:
            raise ValueError("scope and actor are required")
        with self._db() as db:
            row = db.execute("SELECT * FROM queue_scopes WHERE scope=?", (scope,)).fetchone()
            if row:
                if row["operator_config"] != self._json(operator_config):
                    raise QueueConflict("operator configuration is already pinned for this scope")
                return
            if operator_config.revision != 1:
                raise QueueConflict("initial operator configuration revision must be 1")
            db.execute("INSERT INTO queue_scopes VALUES(?,?,?,?,?,?)",
                       (scope, QueuePolicyState.RUNNING.value, 1, 1,
                        self._json(operator_config), self.SCHEMA_VERSION))
            db.execute("INSERT INTO queue_exposure VALUES(?,?,?,?,?)",
                       (scope, 0.0, 0, operator_config.cost_meter_started_at.isoformat(),
                        operator_config.backup_evidence_ref))
            self._append_event(db, self._event(scope=scope, actor=actor, now=now,
                action="scope_initialized", resulting_revision=1,
                configuration_id=str(operator_config.revision),
                evidence_ref=operator_config.backup_evidence_ref))

    def policy(self, scope: str) -> dict:
        with self._connect() as db:
            sr = db.execute("SELECT * FROM queue_scopes WHERE scope=?", (scope,)).fetchone()
            if sr is None:
                raise KeyError(f"queue scope is not configured: {scope}")
            exposure = db.execute("SELECT * FROM queue_exposure WHERE scope=?", (scope,)).fetchone()
        return {"scope": scope, "state": sr["policy_state"],
                "revision": sr["policy_revision"],
                "operator_config": QueueOperatorConfig.model_validate_json(sr["operator_config"]),
                "spent_usd": exposure["spent_usd"] if exposure else 0.0,
                "exposure_updated_at": exposure["updated_at"] if exposure else None}

    def enqueue_batch(self, *, scope: str, intents: list[QueueIntent],
                      batch_idempotency_key: str,
                      operator_config: QueueOperatorConfig,
                      expected_policy_revision: int, actor: str,
                      observed_free_bytes: int, observed_free_inodes: int,
                      capacity_evidence_ref: str, capacity_observed_at: datetime,
                      now: datetime) -> list[QueueEntry]:
        if not scope or not actor or not intents or not batch_idempotency_key:
            raise ValueError("scope, actor, batch idempotency key, and a non-empty finite batch are required")
        if len(batch_idempotency_key) > 200:
            raise ValueError("batch idempotency key cannot exceed 200 characters")
        batch_payload = json.dumps([i.model_dump(mode="json") for i in intents],
                                   sort_keys=True, separators=(",", ":"), allow_nan=False)
        batch_fingerprint = hashlib.sha256(batch_payload.encode("utf-8")).hexdigest()
        keys = [item.idempotency_key for item in intents]
        if len(set(keys)) != len(keys):
            raise QueueAdmissionError("batch contains duplicate idempotency keys")
        with self._db() as db:
            batch_row = db.execute(
                "SELECT fingerprint,entry_ids FROM queue_batches WHERE scope=? AND batch_key=?",
                (scope, batch_idempotency_key)).fetchone()
            if batch_row:
                if batch_row["fingerprint"] != batch_fingerprint:
                    raise QueueAdmissionError("batch idempotency key was already used for a different ordered request list")
                entry_ids = json.loads(batch_row["entry_ids"])
                by_id = {r["entry_id"]: QueueEntry.model_validate_json(r["record"])
                         for r in db.execute("SELECT entry_id,record FROM queue_entries WHERE scope=?",
                                              (scope,)).fetchall()}
                try:
                    return [by_id[item] for item in entry_ids]
                except KeyError as exc:
                    raise QueueConflict("idempotent batch references a missing queue entry") from exc
            if len(intents) > operator_config.max_backlog:
                raise QueueAdmissionError("batch exceeds configured maximum backlog")
            observed_age = (now - capacity_observed_at).total_seconds()
            if observed_age < 0 or observed_age > operator_config.max_capacity_observation_age_seconds:
                raise QueueAdmissionError("fresh enqueue capacity observation is stale or from the future")
            if not capacity_evidence_ref:
                raise ValueError("fresh enqueue capacity evidence reference is required")
            if observed_free_bytes < 0 or observed_free_inodes < 0:
                raise ValueError("measured enqueue capacity cannot be negative")
            scope_row = db.execute("SELECT * FROM queue_scopes WHERE scope=?", (scope,)).fetchone()
            if scope_row is None:
                if expected_policy_revision != 1:
                    raise QueueConflict("initial policy revision must be 1")
                policy_state, policy_revision, next_sequence = QueuePolicyState.RUNNING, 1, 1
                db.execute("INSERT INTO queue_scopes VALUES(?,?,?,?,?,?)",
                           (scope, policy_state.value, policy_revision, next_sequence,
                            self._json(operator_config), self.SCHEMA_VERSION))
                db.execute("INSERT INTO queue_exposure VALUES(?,?,?,?,?)",
                           (scope, 0.0, 0, operator_config.cost_meter_started_at.isoformat(),
                            operator_config.backup_evidence_ref))
            else:
                if scope_row["policy_revision"] != expected_policy_revision:
                    raise QueueConflict("queue policy revision changed")
                if scope_row["operator_config"] != self._json(operator_config):
                    raise QueueConflict("operator configuration is pinned; explicit reconfiguration is required")
                policy_state = QueuePolicyState(scope_row["policy_state"])
                policy_revision = scope_row["policy_revision"]
                next_sequence = scope_row["next_sequence"]

            existing = {}
            for intent in intents:
                row = db.execute("SELECT fingerprint,record FROM queue_entries WHERE scope=? AND idempotency_key=?",
                                 (scope, intent.idempotency_key)).fetchone()
                if row:
                    fingerprint = self._fingerprint(intent)
                    if row["fingerprint"] != fingerprint:
                        raise QueueAdmissionError("idempotency key was already used for different intent")
                    existing[intent.idempotency_key] = QueueEntry.model_validate_json(row["record"])

            new_intents = [x for x in intents if x.idempotency_key not in existing]
            rows = db.execute("SELECT record FROM queue_entries WHERE scope=?", (scope,)).fetchall()
            current_entries = [QueueEntry.model_validate_json(r["record"]) for r in rows]
            if len(current_entries) + len(new_intents) > operator_config.max_backlog:
                raise QueueAdmissionError("queue exceeds configured maximum backlog")
            reserved_cost = sum(e.budget_reserved_usd for e in current_entries)
            exposure = db.execute("SELECT spent_usd FROM queue_exposure WHERE scope=?", (scope,)).fetchone()
            already_spent = exposure[0] if exposure else 0.0
            # Actual occupied files already reduce the fresh statvfs measurement.
            # Keep retained reservation fields for history, but do not count them
            # again after a terminal attempt is reflected in statvfs.
            attempts_by_id = {
                r["attempt_id"]: QueueAttempt.model_validate_json(r["record"])
                for r in db.execute("SELECT attempt_id,record FROM queue_attempts").fetchall()
            }
            live = []
            for entry in current_entries:
                if entry.state in {QueueState.COMPLETED, QueueState.CANCELED, QueueState.FAILED}:
                    continue
                attempt = attempts_by_id.get(entry.active_attempt_id)
                if attempt and attempt.external_outcome == ExternalOutcome.TERMINAL:
                    continue
                live.append(entry)
            reserved_bytes = sum(e.storage_reserved_bytes for e in live)
            reserved_inodes = sum(e.storage_reserved_inodes for e in live)
            batch_cost = sum(i.job.per_run_cost_usd * i.job.max_attempts for i in new_intents)
            batch_bytes = sum(i.job.storage_peak_bytes for i in new_intents)
            batch_inodes = sum(i.job.storage_peak_inodes for i in new_intents)
            if already_spent + reserved_cost + batch_cost > operator_config.campaign_cost_cap_usd:
                raise QueueAdmissionError("batch exceeds authorized cumulative campaign budget")
            if reserved_bytes + batch_bytes + operator_config.reserved_headroom_bytes > observed_free_bytes:
                raise QueueAdmissionError("unknown or insufficient measured free-byte headroom for batch")
            if reserved_inodes + batch_inodes + operator_config.reserved_headroom_inodes > observed_free_inodes:
                raise QueueAdmissionError("insufficient measured free-inode headroom for batch")

            result: dict[str, QueueEntry] = dict(existing)
            for intent in new_intents:
                fingerprint = self._fingerprint(intent)
                entry = QueueEntry(
                    entry_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()), scope=scope,
                    sequence=next_sequence, fingerprint=fingerprint, state=QueueState.WAITING,
                    revision=1, intent=intent,
                    budget_reserved_usd=intent.job.per_run_cost_usd * intent.job.max_attempts,
                    storage_reserved_bytes=intent.job.storage_peak_bytes,
                    storage_reserved_inodes=intent.job.storage_peak_inodes,
                    created_at=now, updated_at=now)
                db.execute("INSERT INTO queue_entries VALUES(?,?,?,?,?,?)",
                           (entry.entry_id, scope, next_sequence, intent.idempotency_key,
                            fingerprint, self._json(entry)))
                event = self._event(scope=scope, entry_id=entry.entry_id, actor=actor,
                                    now=now, action="enqueue", resulting_revision=1,
                                    configuration_id=intent.config_sha256,
                                    runtime_digest=intent.runtime.image_digest,
                                    vm_identity=f"{intent.job.project_id}/{intent.job.zone}/{intent.job.instance_id}",
                                    evidence_ref=capacity_evidence_ref,
                                    detail=(f"Capacity observed at {capacity_observed_at.isoformat()}: "
                                            f"{observed_free_bytes} bytes, {observed_free_inodes} inodes"))
                self._append_event(db, event)
                result[intent.idempotency_key] = entry
                next_sequence += 1
            db.execute("UPDATE queue_scopes SET next_sequence=? WHERE scope=?", (next_sequence, scope))
            ordered_result = [result[i.idempotency_key] for i in intents]
            db.execute("INSERT INTO queue_batches VALUES(?,?,?,?)",
                       (scope, batch_idempotency_key, batch_fingerprint,
                        json.dumps([e.entry_id for e in ordered_result])))
            return ordered_result

    def get_entry(self, entry_id: str) -> QueueEntry | None:
        with self._connect() as db:
            row = db.execute("SELECT record FROM queue_entries WHERE entry_id=?", (entry_id,)).fetchone()
        return QueueEntry.model_validate_json(row[0]) if row else None

    def list_entries(self, scope: str) -> list[QueueEntry]:
        with self._connect() as db:
            rows = db.execute("SELECT record FROM queue_entries WHERE scope=? ORDER BY sequence", (scope,)).fetchall()
        return [QueueEntry.model_validate_json(row[0]) for row in rows]

    def get_attempt(self, attempt_id: str) -> QueueAttempt | None:
        with self._connect() as db:
            row = db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        return QueueAttempt.model_validate_json(row[0]) if row else None

    def list_attempts(self, entry_id: str) -> list[QueueAttempt]:
        with self._connect() as db:
            rows = db.execute("SELECT record FROM queue_attempts WHERE entry_id=? ORDER BY rowid", (entry_id,)).fetchall()
        return [QueueAttempt.model_validate_json(row[0]) for row in rows]

    def list_events(self, scope: str, entry_id: str | None = None) -> list[QueueEvent]:
        with self._connect() as db:
            if entry_id is None:
                rows = db.execute("SELECT record FROM queue_events WHERE scope=? ORDER BY ordinal", (scope,)).fetchall()
            else:
                rows = db.execute("SELECT record FROM queue_events WHERE scope=? AND entry_id=? ORDER BY ordinal",
                                  (scope, entry_id)).fetchall()
        return [QueueEvent.model_validate_json(row[0]) for row in rows]

    def _replace_entry(self, db: sqlite3.Connection, entry: QueueEntry,
                       expected_revision: int) -> QueueEntry:
        entry.revision = expected_revision + 1
        entry.updated_at = entry.updated_at
        result = db.execute("UPDATE queue_entries SET record=? WHERE entry_id=? AND json_extract(record,'$.revision')=?",
                            (self._json(entry), entry.entry_id, expected_revision))
        if result.rowcount != 1:
            raise QueueConflict("entry revision changed")
        return entry

    def claim_next(self, *, scope: str, worker_id: str, lease_seconds: int,
                   boot_id: str, observed_free_bytes: int, observed_free_inodes: int,
                   capacity_evidence_ref: str, capacity_observed_at: datetime,
                   now: datetime) -> QueueClaim | None:
        if lease_seconds <= 0:
            raise ValueError("claim lease must be positive")
        if not boot_id or not capacity_evidence_ref:
            raise ValueError("observed boot identity and capacity evidence are required")
        with self._db() as db:
            sr = self._scope_row(db, scope)
            if QueuePolicyState(sr["policy_state"]) != QueuePolicyState.RUNNING:
                return None
            config = QueueOperatorConfig.model_validate_json(sr["operator_config"])
            age = (now - capacity_observed_at).total_seconds()
            if age < 0 or age > config.max_capacity_observation_age_seconds:
                raise QueueAdmissionError("capacity observation is stale or from the future")
            entries = [QueueEntry.model_validate_json(r[0]) for r in db.execute(
                "SELECT record FROM queue_entries WHERE scope=? ORDER BY sequence", (scope,)).fetchall()]
            if any(e.state in {QueueState.DISPATCHING, QueueState.ACTIVE, QueueState.CANCELING}
                   for e in entries):
                return None
            head = next((e for e in entries if e.state not in {
                QueueState.COMPLETED, QueueState.CANCELED, QueueState.FAILED}), None)
            if head is None or head.state != QueueState.WAITING:
                return None
            if head.next_eligible_at and now < head.next_eligible_at:
                return None
            if (observed_free_bytes < head.intent.job.storage_peak_bytes + config.reserved_headroom_bytes
                    or observed_free_inodes < head.intent.job.storage_peak_inodes + config.reserved_headroom_inodes):
                prior = head.revision
                head.state = QueueState.BLOCKED
                head.blocking_reason = "Fresh VM capacity observation is below job peak plus protected headroom"
                head.last_observation_at = capacity_observed_at
                head.updated_at = now
                self._replace_entry(db, head, prior)
                self._append_event(db, self._event(scope=scope, entry_id=head.entry_id,
                    actor=worker_id, now=now, action="capacity_block", prior_revision=prior,
                    resulting_revision=head.revision, evidence_ref=capacity_evidence_ref,
                    detail=head.blocking_reason))
                return None
            prior = head.revision
            attempt_id, dispatch_id, token = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
            profile = head.intent.job
            attempt = QueueAttempt(
                attempt_id=attempt_id, entry_id=head.entry_id, run_id=head.run_id,
                dispatch_id=dispatch_id, state=AttemptState.DISPATCHING,
                external_outcome=ExternalOutcome.CREATE_INTENT, revision=1,
                worker_id=worker_id, claim_token=token,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                project_id=profile.project_id, zone=profile.zone, instance_id=profile.instance_id,
                boot_id=boot_id, container_name=f"defect-{head.run_id}-{attempt_id}",
                budget_reserved_usd=profile.per_run_cost_usd,
                storage_reserved_bytes=profile.storage_peak_bytes,
                storage_reserved_inodes=profile.storage_peak_inodes,
                created_at=now, updated_at=now)
            db.execute("INSERT INTO queue_attempts VALUES(?,?,?)",
                       (attempt.attempt_id, attempt.entry_id, self._json(attempt)))
            head.active_attempt_id = attempt_id
            head.state = QueueState.DISPATCHING
            head.container_exit_code = None
            head.outputs_verified = None
            head.mlflow_finalized = None
            head.cancellation_requested = False
            head.release_eligible = False
            head.blocking_reason = None
            head.next_eligible_at = None
            head.updated_at = now
            self._replace_entry(db, head, prior)
            self._append_event(db, self._event(
                scope=scope, entry_id=head.entry_id, attempt_id=attempt_id, actor=worker_id,
                now=now, action="claim_create_intent", prior_revision=prior,
                resulting_revision=head.revision, configuration_id=head.intent.config_sha256,
                runtime_digest=head.intent.runtime.image_digest,
                vm_identity=f"{profile.project_id}/{profile.zone}/{profile.instance_id}/{boot_id}",
                evidence_ref=capacity_evidence_ref,
                detail="Create intent persisted before external container side effect"))
            return QueueClaim(entry=head, attempt=attempt)

    def record_dispatch(self, *, entry_id: str, expected_revision: int,
                        attempt_id: str, dispatch_id: str,
                        claim_token: str,
                        external_outcome: ExternalOutcome, actor: str,
                        now: datetime, container_id: str | None = None,
                        process_id: str | None = None, exit_code: int | None = None,
                        log_ref: str | None = None, output_ref: str | None = None,
                        evidence_ref: str | None = None) -> QueueEntry:
        with self._db() as db:
            row = db.execute("SELECT record FROM queue_entries WHERE entry_id=?", (entry_id,)).fetchone()
            if not row:
                raise KeyError(entry_id)
            entry = QueueEntry.model_validate_json(row[0])
            if entry.revision != expected_revision or entry.active_attempt_id != attempt_id:
                raise QueueConflict("entry revision or active attempt changed")
            ar = db.execute("SELECT record FROM queue_attempts WHERE attempt_id=? AND entry_id=?",
                            (attempt_id, entry_id)).fetchone()
            if not ar:
                raise KeyError(attempt_id)
            attempt = QueueAttempt.model_validate_json(ar[0])
            if attempt.dispatch_id != dispatch_id:
                raise QueueConflict("dispatch identity mismatch")
            if attempt.claim_token != claim_token:
                raise QueueConflict("stale worker claim token")
            attempt.external_outcome = external_outcome
            attempt.container_id = container_id or attempt.container_id
            attempt.process_id = process_id or attempt.process_id
            attempt.exit_code = exit_code
            attempt.log_ref = log_ref or attempt.log_ref
            attempt.output_ref = output_ref or attempt.output_ref
            attempt.revision += 1
            attempt.updated_at = now
            if external_outcome == ExternalOutcome.START_INTENT and attempt.start_intent_at is None:
                attempt.start_intent_at = now
                attempt.deadline_at = now + timedelta(seconds=entry.intent.job.max_runtime_seconds)
            if external_outcome in {ExternalOutcome.UNKNOWN, ExternalOutcome.CREATE_INTENT,
                                    ExternalOutcome.START_INTENT}:
                attempt.state = AttemptState.UNKNOWN
                entry.state = QueueState.BLOCKED
                entry.blocking_reason = "External container outcome is uncertain; reconcile identity before retry"
            elif external_outcome in {ExternalOutcome.CREATED, ExternalOutcome.STARTED}:
                attempt.state = AttemptState.ACTIVE
                if external_outcome == ExternalOutcome.STARTED and attempt.started_at is None:
                    attempt.started_at = now
                    if attempt.deadline_at is None:
                        attempt.deadline_at = now + timedelta(seconds=entry.intent.job.max_runtime_seconds)
                entry.state = QueueState.ACTIVE
                entry.blocking_reason = None
            elif external_outcome == ExternalOutcome.TERMINAL:
                attempt.state = AttemptState.TERMINAL
                entry.container_exit_code = exit_code
                entry.state = QueueState.BLOCKED
                entry.blocking_reason = "Container termination observed; output verification and finalization are separate"
            prior = entry.revision
            entry.updated_at = now
            self._replace_entry(db, entry, prior)
            db.execute("UPDATE queue_attempts SET record=? WHERE attempt_id=?",
                       (self._json(attempt), attempt_id))
            self._append_event(db, self._event(
                scope=entry.scope, entry_id=entry_id, attempt_id=attempt_id, actor=actor,
                now=now, action=f"dispatch_{external_outcome.value}", prior_revision=prior,
                resulting_revision=entry.revision, configuration_id=entry.intent.config_sha256,
                runtime_digest=entry.intent.runtime.image_digest,
                vm_identity=f"{attempt.project_id}/{attempt.zone}/{attempt.instance_id}/{attempt.boot_id}",
                container_identity=container_id, process_identity=process_id,
                log_ref=log_ref, evidence_ref=evidence_ref))
            return entry

    def transition(self, *, entry_id: str, expected_revision: int, state: QueueState,
                   actor: str, action: str, now: datetime,
                   blocking_reason: str | None = None,
                   next_eligible_at: datetime | None = None,
                   evidence_ref: str | None = None) -> QueueEntry:
        with self._db() as db:
            row = db.execute("SELECT record FROM queue_entries WHERE entry_id=?", (entry_id,)).fetchone()
            if not row:
                raise KeyError(entry_id)
            entry = QueueEntry.model_validate_json(row[0])
            if entry.revision != expected_revision:
                raise QueueConflict("entry revision changed")
            if state == QueueState.WAITING and entry.state == QueueState.BLOCKED and entry.active_attempt_id:
                attempt = db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?",
                                     (entry.active_attempt_id,)).fetchone()
                if attempt and QueueAttempt.model_validate_json(attempt[0]).external_outcome in {
                    ExternalOutcome.UNKNOWN, ExternalOutcome.CREATE_INTENT, ExternalOutcome.START_INTENT,
                    ExternalOutcome.CREATED, ExternalOutcome.STARTED
                }:
                    raise QueueConflict("cannot make unresolved external work retryable")
            if state == QueueState.CANCELED:
                if entry.state not in {QueueState.WAITING, QueueState.BLOCKED, QueueState.CANCELING}:
                    raise QueueConflict("active work must be observed terminal before cancellation")
                if entry.active_attempt_id:
                    arow = db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?",
                                      (entry.active_attempt_id,)).fetchone()
                    if arow:
                        attempt = QueueAttempt.model_validate_json(arow[0])
                        if attempt.external_outcome != ExternalOutcome.TERMINAL:
                            raise QueueConflict("active work must be observed terminal before cancellation")
                entry.budget_reserved_usd = 0
                # Retain storage evidence for any attempt that may have written files.
                if not entry.active_attempt_id:
                    entry.storage_reserved_bytes = 0
                    entry.storage_reserved_inodes = 0
            active_row = (db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?",
                                     (entry.active_attempt_id,)).fetchone()
                          if entry.active_attempt_id else None)
            active_attempt = QueueAttempt.model_validate_json(active_row[0]) if active_row else None
            entry.state, entry.blocking_reason, entry.updated_at = state, blocking_reason, now
            if state == QueueState.CANCELING:
                entry.cancellation_requested = True
            entry.next_eligible_at = next_eligible_at
            self._replace_entry(db, entry, expected_revision)
            self._append_event(db, self._event(
                scope=entry.scope, entry_id=entry_id, actor=actor, now=now, action=action,
                prior_revision=expected_revision, resulting_revision=entry.revision,
                configuration_id=entry.intent.config_sha256,
                runtime_digest=entry.intent.runtime.image_digest,
                vm_identity=(f"{active_attempt.project_id}/{active_attempt.zone}/{active_attempt.instance_id}/"
                             f"{active_attempt.boot_id}" if active_attempt else
                             f"{entry.intent.job.project_id}/{entry.intent.job.zone}/{entry.intent.job.instance_id}"),
                container_identity=active_attempt.container_id if active_attempt else None,
                process_identity=active_attempt.process_id if active_attempt else None,
                log_ref=active_attempt.log_ref if active_attempt else None,
                evidence_ref=evidence_ref,
                detail=blocking_reason))
            return entry

    def set_policy(self, *, scope: str, expected_policy_revision: int,
                   state: QueuePolicyState, actor: str, now: datetime) -> int:
        with self._db() as db:
            row = self._scope_row(db, scope)
            if row["policy_revision"] != expected_policy_revision:
                raise QueueConflict("queue policy revision changed")
            revision = expected_policy_revision + 1
            db.execute("UPDATE queue_scopes SET policy_state=?,policy_revision=? WHERE scope=?",
                       (state.value, revision, scope))
            self._append_event(db, self._event(scope=scope, actor=actor, now=now,
                                               action=f"policy_{state.value}",
                                               prior_revision=expected_policy_revision,
                                               resulting_revision=revision))
            return revision

    def cancel_waiting(self, *, entry_id: str, expected_revision: int,
                       actor: str, now: datetime) -> QueueEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        if entry.state != QueueState.WAITING:
            raise QueueConflict("only waiting entries can be atomically removed")
        return self.transition(entry_id=entry_id, expected_revision=expected_revision,
                               state=QueueState.CANCELED, actor=actor, action="cancel_waiting", now=now)

    def request_active_cancel(self, *, entry_id: str, expected_revision: int,
                              actor: str, now: datetime) -> QueueEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        if (entry.state not in {QueueState.DISPATCHING, QueueState.ACTIVE, QueueState.BLOCKED}
                or not entry.active_attempt_id):
            raise QueueConflict("entry has no active attempt to cancel")
        return self.transition(entry_id=entry_id, expected_revision=expected_revision,
                               state=QueueState.CANCELING, actor=actor,
                               action="request_active_cancel", now=now,
                               blocking_reason="Cancellation requested; slot remains held until observed terminal")

    def retry(self, *, entry_id: str, expected_revision: int, actor: str,
              now: datetime, evidence_ref: str,
              classification: RetryClassification = RetryClassification.UNKNOWN) -> QueueEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        attempts = self.list_attempts(entry_id)
        if not attempts:
            raise QueueConflict("entry has no attempt eligible for retry")
        last = attempts[-1]
        confirmed_failure = last.state == AttemptState.FAILED and last.exit_code not in (None, 0)
        verified_interruption = last.state == AttemptState.INTERRUPTED and last.external_outcome == ExternalOutcome.TERMINAL
        if not (confirmed_failure or verified_interruption):
            raise QueueConflict("only a confirmed failed or verified interrupted terminal attempt can be retried")
        classification = RetryClassification(classification)
        eligible_class = (RetryClassification.INTERRUPTED if verified_interruption
                          else RetryClassification.TRANSIENT)
        if classification != eligible_class or not evidence_ref:
            raise QueueAdmissionError("retry requires an authorized recoverability classification and evidence")
        if len(attempts) >= entry.intent.job.max_attempts:
            raise QueueAdmissionError("configured maximum attempts exhausted")
        if entry.budget_reserved_usd < entry.intent.job.per_run_cost_usd:
            raise QueueAdmissionError("no remaining per-run budget reservation for another attempt")
        policy = self.policy(entry.scope)
        config = policy["operator_config"]
        eligible_at = attempts[-1].updated_at + timedelta(seconds=config.retry_backoff_seconds)
        entry = self.transition(entry_id=entry_id, expected_revision=expected_revision,
                                state=QueueState.WAITING, actor=actor,
                                action="authorized_retry", now=now,
                                next_eligible_at=eligible_at, evidence_ref=evidence_ref)
        return entry

    def mark_interrupted(self, *, entry_id: str, expected_revision: int,
                         actor: str, now: datetime, evidence_ref: str) -> QueueEntry:
        if not evidence_ref:
            raise ValueError("verified interruption evidence is required")
        with self._db() as db:
            row = db.execute("SELECT record FROM queue_entries WHERE entry_id=?", (entry_id,)).fetchone()
            if not row:
                raise KeyError(entry_id)
            entry = QueueEntry.model_validate_json(row[0])
            if entry.revision != expected_revision or not entry.active_attempt_id:
                raise QueueConflict("entry revision or active attempt changed")
            attempt_row = db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?",
                                     (entry.active_attempt_id,)).fetchone()
            if not attempt_row:
                raise QueueConflict("active attempt record is missing")
            attempt = QueueAttempt.model_validate_json(attempt_row[0])
            if attempt.external_outcome == ExternalOutcome.TERMINAL:
                raise QueueConflict("attempt is already terminal")
            attempt.external_outcome = ExternalOutcome.TERMINAL
            attempt.state = AttemptState.INTERRUPTED
            attempt.exit_code = None
            attempt.revision += 1
            attempt.updated_at = now
            entry.state = QueueState.BLOCKED
            entry.container_exit_code = None
            entry.outputs_verified = None
            entry.mlflow_finalized = None
            entry.blocking_reason = "Training was verified interrupted; explicit retry or skip is required"
            attempt_count = db.execute("SELECT count(*) FROM queue_attempts WHERE entry_id=?",
                                       (entry_id,)).fetchone()[0]
            remaining_attempts = max(0, entry.intent.job.max_attempts - attempt_count)
            entry.budget_reserved_usd = entry.intent.job.per_run_cost_usd * remaining_attempts
            entry.updated_at = now
            self._replace_entry(db, entry, expected_revision)
            db.execute("UPDATE queue_attempts SET record=? WHERE attempt_id=?",
                       (self._json(attempt), attempt.attempt_id))
            policy_row = self._scope_row(db, entry.scope)
            policy_revision = policy_row["policy_revision"] + 1
            db.execute("UPDATE queue_scopes SET policy_state=?,policy_revision=? WHERE scope=?",
                       (QueuePolicyState.PAUSED.value, policy_revision, entry.scope))
            self._append_event(db, self._event(
                scope=entry.scope, entry_id=entry_id, attempt_id=attempt.attempt_id, actor=actor,
                now=now, action="attempt_verified_interrupted", prior_revision=expected_revision,
                resulting_revision=entry.revision,
                configuration_id=entry.intent.config_sha256,
                runtime_digest=entry.intent.runtime.image_digest,
                vm_identity=f"{attempt.project_id}/{attempt.zone}/{attempt.instance_id}/{attempt.boot_id}",
                container_identity=attempt.container_id, process_identity=attempt.process_id,
                log_ref=attempt.log_ref, evidence_ref=evidence_ref,
                detail="No exit code inferred; outputs and finalization remain unverified"))
            self._append_event(db, self._event(
                scope=entry.scope, entry_id=entry_id, attempt_id=attempt.attempt_id, actor=actor,
                now=now, action="interruption_paused_queue",
                prior_revision=policy_row["policy_revision"], resulting_revision=policy_revision,
                evidence_ref=evidence_ref,
                detail="Recovery requires explicit operator action"))
            return entry

    def skip(self, *, entry_id: str, expected_revision: int, actor: str,
             now: datetime, evidence_ref: str) -> QueueEntry:
        return self.transition(entry_id=entry_id, expected_revision=expected_revision,
                               state=QueueState.CANCELED, actor=actor,
                               action="authorized_skip", now=now, evidence_ref=evidence_ref)

    def reconcile_unknown(self, *, entry_id: str, expected_revision: int,
                          resolution: ExternalOutcome, actor: str, now: datetime,
                          evidence_ref: str) -> QueueEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        if entry.state not in {QueueState.BLOCKED, QueueState.CANCELING} or not entry.active_attempt_id:
            raise QueueConflict("entry has no unresolved attempt")
        attempt = self.get_attempt(entry.active_attempt_id)
        if attempt is None:
            raise QueueConflict("attempt identity is missing")
        # Reconciliation records evidence; it deliberately does not reset to waiting.
        return self.record_dispatch(entry_id=entry_id, expected_revision=expected_revision,
                                    attempt_id=attempt.attempt_id, dispatch_id=attempt.dispatch_id,
                                    claim_token=attempt.claim_token,
                                    external_outcome=resolution, actor=actor, now=now,
                                    container_id=attempt.container_id, process_id=attempt.process_id,
                                    exit_code=attempt.exit_code, log_ref=attempt.log_ref,
                                    output_ref=attempt.output_ref, evidence_ref=evidence_ref)

    def record_finalization(self, *, entry_id: str, expected_revision: int,
                            outputs_verified: bool, mlflow_finalized: bool,
                            actor: str, now: datetime, evidence_ref: str) -> QueueEntry:
        with self._db() as db:
            row = db.execute("SELECT record FROM queue_entries WHERE entry_id=?", (entry_id,)).fetchone()
            if not row:
                raise KeyError(entry_id)
            entry = QueueEntry.model_validate_json(row[0])
            if entry.revision != expected_revision:
                raise QueueConflict("entry revision changed")
            active_row = (db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?",
                                     (entry.active_attempt_id,)).fetchone()
                          if entry.active_attempt_id else None)
            active = QueueAttempt.model_validate_json(active_row[0]) if active_row else None
            if (entry.container_exit_code is None and not (
                    entry.cancellation_requested and active is not None
                    and active.external_outcome == ExternalOutcome.TERMINAL)):
                raise QueueConflict("container termination must be observed before finalization")
            entry.outputs_verified = outputs_verified
            entry.mlflow_finalized = mlflow_finalized
            entry.release_eligible = False
            attempt_row = (db.execute("SELECT record FROM queue_attempts WHERE attempt_id=?",
                                      (entry.active_attempt_id,)).fetchone()
                           if entry.active_attempt_id else None)
            attempt = QueueAttempt.model_validate_json(attempt_row[0]) if attempt_row else None
            scope_row = self._scope_row(db, entry.scope)
            config = QueueOperatorConfig.model_validate_json(scope_row["operator_config"])
            attempt_count = db.execute("SELECT count(*) FROM queue_attempts WHERE entry_id=?",
                                       (entry_id,)).fetchone()[0]
            if entry.cancellation_requested:
                if mlflow_finalized:
                    entry.state, entry.blocking_reason = QueueState.CANCELED, None
                    entry.budget_reserved_usd = 0
                    if attempt:
                        attempt.state = AttemptState.CANCELED
                else:
                    entry.state = QueueState.BLOCKED
                    entry.blocking_reason = "Cancellation termination observed; tracking finalization remains unverified"
            elif outputs_verified and mlflow_finalized and entry.container_exit_code == 0:
                entry.state, entry.blocking_reason = QueueState.COMPLETED, None
                entry.budget_reserved_usd = 0
                if attempt:
                    attempt.state = AttemptState.SUCCEEDED
            elif entry.container_exit_code != 0 and not mlflow_finalized:
                entry.state = QueueState.BLOCKED
                entry.blocking_reason = "Tracking finalization is unverified; queue continuation is held"
                remaining_attempts = max(0, entry.intent.job.max_attempts - attempt_count)
                entry.budget_reserved_usd = entry.intent.job.per_run_cost_usd * remaining_attempts
                # Keep the attempt terminal, not failed/retryable, until the
                # tracking outcome has been explicitly reconciled.
            elif entry.container_exit_code != 0 and config.failure_policy.value == "continue":
                entry.state = QueueState.FAILED
                entry.blocking_reason = "Confirmed training failure; configured continuation policy advanced the queue"
                entry.budget_reserved_usd = 0
                if attempt:
                    attempt.state = AttemptState.FAILED
            elif entry.container_exit_code != 0:
                entry.state = QueueState.BLOCKED
                entry.blocking_reason = "Confirmed training failure; queue paused for authorized retry or skip"
                remaining_attempts = max(0, entry.intent.job.max_attempts - attempt_count)
                entry.budget_reserved_usd = entry.intent.job.per_run_cost_usd * remaining_attempts
                if attempt:
                    attempt.state = AttemptState.FAILED
                policy_revision = scope_row["policy_revision"] + 1
                db.execute("UPDATE queue_scopes SET policy_state=?,policy_revision=? WHERE scope=?",
                           (QueuePolicyState.PAUSED.value, policy_revision, entry.scope))
                self._append_event(db, self._event(
                    scope=entry.scope, entry_id=entry_id, attempt_id=attempt.attempt_id if attempt else None,
                    actor=actor, now=now, action="confirmed_failure_paused_queue",
                    prior_revision=scope_row["policy_revision"], resulting_revision=policy_revision,
                    configuration_id=entry.intent.config_sha256,
                    runtime_digest=entry.intent.runtime.image_digest,
                    evidence_ref=evidence_ref,
                    detail="Default failure policy requires explicit operator resume or skip"))
            else:
                entry.state = QueueState.BLOCKED
                entry.blocking_reason = "Completion gate failed; inspect output and tracking evidence"
                entry.budget_reserved_usd = 0
                if attempt:
                    attempt.state = AttemptState.SUCCEEDED
            if attempt:
                attempt.revision += 1
                attempt.updated_at = now
                db.execute("UPDATE queue_attempts SET record=? WHERE attempt_id=?",
                           (self._json(attempt), attempt.attempt_id))
            entry.updated_at = now
            self._replace_entry(db, entry, expected_revision)
            self._append_event(db, self._event(
                scope=entry.scope, entry_id=entry_id, actor=actor, now=now,
                action="finalization_recorded", prior_revision=expected_revision,
                resulting_revision=entry.revision,
                configuration_id=entry.intent.config_sha256,
                runtime_digest=entry.intent.runtime.image_digest,
                vm_identity=(f"{attempt.project_id}/{attempt.zone}/{attempt.instance_id}/{attempt.boot_id}"
                             if attempt else None),
                container_identity=attempt.container_id if attempt else None,
                process_identity=attempt.process_id if attempt else None,
                log_ref=attempt.log_ref if attempt else None,
                evidence_ref=evidence_ref, detail=entry.blocking_reason))
            return entry

    def record_exposure(self, *, scope: str, expected_policy_revision: int,
                        additional_cost_usd: float, actor: str, now: datetime,
                        evidence_ref: str) -> float:
        if additional_cost_usd < 0:
            raise ValueError("cumulative exposure increments cannot be negative")
        with self._db() as db:
            sr = self._scope_row(db, scope)
            if sr["policy_revision"] != expected_policy_revision:
                raise QueueConflict("queue policy revision changed")
            config = QueueOperatorConfig.model_validate_json(sr["operator_config"])
            row = db.execute("SELECT spent_usd,revision FROM queue_exposure WHERE scope=?", (scope,)).fetchone()
            spent, revision = (row[0], row[1]) if row else (0.0, 0)
            active = db.execute("SELECT record FROM queue_entries WHERE scope=?", (scope,)).fetchall()
            reserved = sum(e.budget_reserved_usd for r in active
                           if (e := QueueEntry.model_validate_json(r[0])).state not in {
                               QueueState.COMPLETED, QueueState.CANCELED, QueueState.FAILED})
            total = spent + additional_cost_usd
            db.execute("INSERT INTO queue_exposure VALUES(?,?,?,?,?) ON CONFLICT(scope) DO UPDATE SET "
                       "spent_usd=excluded.spent_usd,revision=excluded.revision,updated_at=excluded.updated_at,"
                       "last_evidence_ref=excluded.last_evidence_ref",
                       (scope, total, revision + 1, now.isoformat(), evidence_ref))
            policy_revision = expected_policy_revision
            if total + reserved > config.campaign_cost_cap_usd:
                policy_revision += 1
                db.execute("UPDATE queue_scopes SET policy_state=?,policy_revision=? WHERE scope=?",
                           (QueuePolicyState.PAUSED.value, policy_revision, scope))
            self._append_event(db, self._event(scope=scope, actor=actor, now=now,
                action="vm_exposure_accounted", prior_revision=revision,
                resulting_revision=revision + 1, evidence_ref=evidence_ref,
                detail=f"Cumulative accounted VM/queue exposure is ${total:.6f}"))
            if policy_revision != expected_policy_revision:
                self._append_event(db, self._event(scope=scope, actor=actor, now=now,
                    action="campaign_cap_exceeded_pause", prior_revision=expected_policy_revision,
                    resulting_revision=policy_revision, evidence_ref=evidence_ref,
                    detail="Recorded liability exceeded the campaign cap; dispatch paused"))
            return total

    def account_until(self, *, scope: str, expected_policy_revision: int,
                      observed_at: datetime, actor: str, evidence_ref: str) -> float:
        if not evidence_ref:
            raise ValueError("cost observation evidence is required")
        with self._db() as db:
            sr = self._scope_row(db, scope)
            if sr["policy_revision"] != expected_policy_revision:
                raise QueueConflict("queue policy revision changed")
            config = QueueOperatorConfig.model_validate_json(sr["operator_config"])
            row = db.execute("SELECT spent_usd,revision,updated_at FROM queue_exposure WHERE scope=?",
                             (scope,)).fetchone()
            if row is None:
                raise QueueConflict("cost meter is not initialized")
            last_at = datetime.fromisoformat(row["updated_at"])
            if observed_at < last_at:
                raise QueueConflict("cost observation is older than the persisted meter cursor")
            delta = (observed_at - last_at).total_seconds()
            amount = delta * config.vm_hourly_cost_usd / 3600.0
            total = row["spent_usd"] + amount
            active = db.execute("SELECT record FROM queue_entries WHERE scope=?", (scope,)).fetchall()
            reserved = sum(e.budget_reserved_usd for r in active
                           if (e := QueueEntry.model_validate_json(r[0])).state not in {
                               QueueState.COMPLETED, QueueState.CANCELED, QueueState.FAILED})
            db.execute("UPDATE queue_exposure SET spent_usd=?,revision=?,updated_at=?,last_evidence_ref=? WHERE scope=?",
                       (total, row["revision"] + 1, observed_at.isoformat(), evidence_ref, scope))
            resulting_policy_revision = expected_policy_revision
            if total + reserved > config.campaign_cost_cap_usd:
                resulting_policy_revision += 1
                db.execute("UPDATE queue_scopes SET policy_state=?,policy_revision=? WHERE scope=?",
                           (QueuePolicyState.PAUSED.value, resulting_policy_revision, scope))
            self._append_event(db, self._event(scope=scope, actor=actor, now=observed_at,
                action="vm_uptime_exposure_accounted", prior_revision=row["revision"],
                resulting_revision=row["revision"] + 1, evidence_ref=evidence_ref,
                detail=f"Metered {delta:.3f}s; cumulative VM/queue exposure is ${total:.6f}"))
            if resulting_policy_revision != expected_policy_revision:
                self._append_event(db, self._event(scope=scope, actor=actor, now=observed_at,
                    action="campaign_cap_exceeded_pause", prior_revision=expected_policy_revision,
                    resulting_revision=resulting_policy_revision, evidence_ref=evidence_ref,
                    detail="Recorded liability exceeded the campaign cap; dispatch paused"))
            return total

    def backup(self, destination: str) -> str:
        """Use SQLite's online backup API for a transactionally consistent copy."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            raise FileExistsError("backup destination must be a new, non-symlink path")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags, 0o600)
        os.close(fd)
        source_db = self._connect()
        backup_db = sqlite3.connect(str(target))
        try:
            source_db.backup(backup_db)
            integrity = backup_db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(f"backup integrity check failed: {integrity}")
            backup_db.commit()
        finally:
            backup_db.close()
            source_db.close()
        return str(target)

    def verify_restore(self, source: str) -> QueueSnapshot:
        """Validate schema and records from a backup without activating it."""
        db = sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            check = db.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise ValueError(f"backup integrity check failed: {check}")
            scopes = db.execute("SELECT * FROM queue_scopes").fetchall()
            if len(scopes) != 1:
                raise ValueError("backup must contain exactly one queue scope")
            scope_row = scopes[0]
            if scope_row["schema_version"] != self.SCHEMA_VERSION:
                raise ValueError("unsupported queue schema version")
            entries = [QueueEntry.model_validate_json(r[0]) for r in db.execute(
                "SELECT record FROM queue_entries WHERE scope=? ORDER BY sequence", (scope_row["scope"],)).fetchall()]
            attempts = [QueueAttempt.model_validate_json(r[0]) for r in db.execute(
                "SELECT record FROM queue_attempts ORDER BY rowid").fetchall()]
            events = [QueueEvent.model_validate_json(r[0]) for r in db.execute(
                "SELECT record FROM queue_events WHERE scope=? ORDER BY ordinal", (scope_row["scope"],)).fetchall()]
            entry_by_id = {entry.entry_id: entry for entry in entries}
            attempt_by_id = {attempt.attempt_id: attempt for attempt in attempts}
            if len(entry_by_id) != len(entries) or len(attempt_by_id) != len(attempts):
                raise ValueError("backup contains duplicate queue record identities")
            for attempt in attempts:
                entry = entry_by_id.get(attempt.entry_id)
                if entry is None or entry.run_id != attempt.run_id:
                    raise ValueError("backup attempt references a missing or mismatched entry/run")
            for entry in entries:
                if entry.active_attempt_id:
                    attempt = attempt_by_id.get(entry.active_attempt_id)
                    if attempt is None or attempt.entry_id != entry.entry_id:
                        raise ValueError("backup entry references a missing or mismatched active attempt")
            for event in events:
                if event.entry_id and event.entry_id not in entry_by_id:
                    raise ValueError("backup event references a missing queue entry")
                if event.attempt_id and event.attempt_id not in attempt_by_id:
                    raise ValueError("backup event references a missing queue attempt")
                if event.attempt_id and event.entry_id and attempt_by_id[event.attempt_id].entry_id != event.entry_id:
                    raise ValueError("backup event attempt and entry identities do not match")
            for batch in db.execute("SELECT entry_ids FROM queue_batches WHERE scope=?",
                                    (scope_row["scope"],)).fetchall():
                batch_ids = json.loads(batch[0])
                if any(entry_id not in entry_by_id for entry_id in batch_ids):
                    raise ValueError("backup batch references a missing queue entry")
            for batch in db.execute("SELECT fingerprint,entry_ids FROM queue_batches WHERE scope=?",
                                    (scope_row["scope"],)).fetchall():
                batch_entries = [entry_by_id[entry_id] for entry_id in json.loads(batch["entry_ids"])]
                batch_payload = json.dumps([e.intent.model_dump(mode="json") for e in batch_entries],
                                           sort_keys=True, separators=(",", ":"), allow_nan=False)
                if hashlib.sha256(batch_payload.encode("utf-8")).hexdigest() != batch["fingerprint"]:
                    raise ValueError("backup batch request fingerprint does not match its entries")
            return QueueSnapshot(schema_version=self.SCHEMA_VERSION, scope=scope_row["scope"],
                                 policy_state=QueuePolicyState(scope_row["policy_state"]),
                                 policy_revision=scope_row["policy_revision"],
                                 operator_config=QueueOperatorConfig.model_validate_json(scope_row["operator_config"]),
                                 entries=entries, attempts=attempts, events=events)
        finally:
            db.close()
