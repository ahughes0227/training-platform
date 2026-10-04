"""Durable control-plane state. SQLite is useful locally; production stores are injectable."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote_plus

from defect_platform.contracts import RunRecord, RunState
from defect_platform.control.setup import IntakeSession


class IntakeSessionConflictError(ValueError):
    """The caller wrote against an old session revision."""

    def __init__(self, message: str, current: IntakeSession | None = None):
        super().__init__(message)
        self.current = current


class IntakeSessionExpiredError(KeyError):
    """The server-side draft is no longer available because its TTL elapsed."""


class IntakeSessionStore(Protocol):
    def create(self, session: IntakeSession) -> IntakeSession: ...
    def get(self, session_id: str, *, now: datetime | None = None) -> IntakeSession | None: ...
    def update(self, session: IntakeSession, expected_revision: int) -> IntakeSession: ...
    def delete_expired(self, *, now: datetime | None = None) -> int: ...


@dataclass(frozen=True)
class EventRecord:
    """An append-only run event with a per-run monotonic cursor."""

    run_id: str
    sequence: int
    event_id: str
    event_type: str
    payload: dict
    created_at: str

    def model_dump(self) -> dict:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "created_at": self.created_at,
        }


def compact_status(run: RunRecord) -> dict[str, Any]:
    """Return the small, safe-by-default status view used by API clients."""
    blocker = None
    next_action = None
    if run.state is RunState.FAILED:
        blocker = run.failure_code or run.failure_message or "run failed"
        next_action = "review the failure and resubmit after remediation"
    elif run.state in {RunState.PENDING, RunState.PREPARING, RunState.SUBMITTED, RunState.RUNNING}:
        next_action = "wait for the next durable run event"
    elif run.state is RunState.SUCCEEDED:
        next_action = "review metrics and approve a release if appropriate"
    return {
        "run_id": run.run_id,
        "object_slug": run.object_slug,
        "experiment_id": run.experiment_id,
        "dataset_version_id": run.dataset_version_id,
        "runtime_id": run.runtime_id,
        "state": run.state.value,
        "created_at": run.created_at.isoformat(),
        "vertex_job_name": run.vertex_job_name,
        "mlflow_run_id": run.mlflow_run_id,
        "failure_code": run.failure_code,
        "failure_message": run.failure_message,
        "logs_uri": run.logs_uri,
        "blocker": blocker,
        "next_action": next_action,
        "links": {"logs": run.logs_uri} if run.logs_uri else {},
    }


def _session_now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)).astimezone(UTC)


class SQLiteIntakeSessionStore:
    """SQLite persistence for setup sessions with an atomic revision fence."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS intake_sessions (
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                schema_ref TEXT NOT NULL, draft TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL)""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _decode(row) -> IntakeSession:
        return IntakeSession.model_validate(
            {
                "session_id": row["session_id"],
                "revision": row["revision"],
                "schema_ref": row["schema_ref"],
                "draft": json.loads(row["draft"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "expires_at": row["expires_at"],
            }
        )

    def create(self, session: IntakeSession) -> IntakeSession:
        with self._db() as db:
            try:
                db.execute(
                    "INSERT INTO intake_sessions VALUES (?,?,?,?,?,?,?)",
                    (
                        session.session_id,
                        session.revision,
                        session.schema_ref,
                        session.draft.model_dump_json(),
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                        session.expires_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"intake session already exists: {session.session_id}") from exc
        return session

    def get(self, session_id: str, *, now: datetime | None = None) -> IntakeSession | None:
        with self._db() as db:
            row = db.execute(
                "SELECT * FROM intake_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        session = self._decode(row)
        if session.expires_at <= _session_now(now):
            self.delete_expired(now=now)
            return None
        return session

    def update(self, session: IntakeSession, expected_revision: int) -> IntakeSession:
        now = _session_now()
        if session.revision != expected_revision + 1:
            raise IntakeSessionConflictError(
                f"intake session revision conflict: expected {expected_revision}, submitted {session.revision}"
            )
        with self._db() as db:
            result = db.execute(
                """UPDATE intake_sessions SET revision=?,schema_ref=?,draft=?,
                updated_at=?,expires_at=? WHERE session_id=? AND revision=? AND expires_at>?""",
                (
                    session.revision,
                    session.schema_ref,
                    session.draft.model_dump_json(),
                    session.updated_at.isoformat(),
                    session.expires_at.isoformat(),
                    session.session_id,
                    expected_revision,
                    now.isoformat(),
                ),
            )
            if result.rowcount == 1:
                return session
            row = db.execute(
                "SELECT * FROM intake_sessions WHERE session_id=?", (session.session_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"intake session not found: {session.session_id}")
        current = self._decode(row)
        if current.expires_at <= now:
            self.delete_expired(now=now)
            raise IntakeSessionExpiredError(session.session_id)
        raise IntakeSessionConflictError(
            f"intake session revision conflict: expected {expected_revision}, current {current.revision}",
            current,
        )

    def delete_expired(self, *, now: datetime | None = None) -> int:
        with self._db() as db:
            return db.execute(
                "DELETE FROM intake_sessions WHERE expires_at<=?", (_session_now(now).isoformat(),)
            ).rowcount


class RunStore(Protocol):
    def create(
        self, run: RunRecord, request_fingerprint: str, payload: dict
    ) -> tuple[RunRecord, bool]: ...
    def get(self, run_id: str) -> RunRecord | None: ...
    def by_key(self, key: str) -> RunRecord | None: ...
    def get_payload(self, run_id: str) -> dict | None: ...
    def update(self, run: RunRecord) -> None: ...
    def update_with_event(
        self,
        run: RunRecord,
        event_type: str,
        payload: dict[str, Any],
        *,
        event_id: str | None = None,
    ) -> None: ...
    def list(self, object_slug: str | None = None) -> list[RunRecord]: ...
    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
        *,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> EventRecord: ...
    def read_events(
        self, run_id: str, *, cursor: int = 0, limit: int = 100
    ) -> tuple[list[EventRecord], int | None]: ...
    def compact(self, run_id: str) -> dict[str, Any] | None: ...


class SQLiteRunStore:
    """SQLite-backed store with unique idempotency keys and atomic record updates."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                fingerprint TEXT NOT NULL, payload TEXT NOT NULL, record TEXT NOT NULL,
                created_at TEXT NOT NULL, projection TEXT)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)").fetchall()}
            if "projection" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN projection TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS run_events (
                run_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                event_id TEXT NOT NULL, event_type TEXT NOT NULL,
                payload TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(run_id, sequence), UNIQUE(run_id, event_id))""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(
        self, run: RunRecord, request_fingerprint: str, payload: dict
    ) -> tuple[RunRecord, bool]:
        key = str(payload["idempotency_key"])
        with self._db() as db:
            row = db.execute(
                "SELECT fingerprint,record FROM runs WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row:
                if row["fingerprint"] != request_fingerprint:
                    raise ValueError("idempotency key was already used for a different request")
                return RunRecord.model_validate_json(row["record"]), False
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    key,
                    request_fingerprint,
                    json.dumps(payload, sort_keys=True),
                    run.model_dump_json(),
                    run.created_at.isoformat(),
                    json.dumps(compact_status(run), sort_keys=True),
                ),
            )
            db.execute(
                "INSERT INTO run_events VALUES (?,?,?,?,?,?)",
                (
                    run.run_id,
                    1,
                    f"{run.run_id}:run.created",
                    "run.created",
                    json.dumps({"state": run.state.value}),
                    run.created_at.isoformat(),
                ),
            )
        return run, True

    def get(self, run_id: str) -> RunRecord | None:
        with self._db() as db:
            row = db.execute("SELECT record FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row["record"]) if row else None

    def by_key(self, key: str) -> RunRecord | None:
        with self._db() as db:
            row = db.execute("SELECT record FROM runs WHERE idempotency_key=?", (key,)).fetchone()
        return RunRecord.model_validate_json(row["record"]) if row else None

    def get_payload(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def update(self, run: RunRecord) -> None:
        with self._db() as db:
            result = db.execute(
                "UPDATE runs SET record=?,projection=? WHERE run_id=?",
                (
                    run.model_dump_json(),
                    json.dumps(compact_status(run), sort_keys=True),
                    run.run_id,
                ),
            )
            if result.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")

    def update_with_event(
        self,
        run: RunRecord,
        event_type: str,
        payload: dict[str, Any],
        *,
        event_id: str | None = None,
    ) -> None:
        event_id = event_id or str(uuid.uuid4())
        with self._db() as db:
            result = db.execute(
                "UPDATE runs SET record=?,projection=? WHERE run_id=?",
                (
                    run.model_dump_json(),
                    json.dumps(compact_status(run), sort_keys=True),
                    run.run_id,
                ),
            )
            if result.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")
            sequence = int(
                db.execute(
                    "SELECT COALESCE(MAX(sequence),0)+1 FROM run_events WHERE run_id=?",
                    (run.run_id,),
                ).fetchone()[0]
            )
            db.execute(
                "INSERT INTO run_events VALUES (?,?,?,?,?,?)",
                (
                    run.run_id,
                    sequence,
                    event_id,
                    event_type,
                    json.dumps(payload, sort_keys=True),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        query, args = "SELECT record FROM runs", ()
        if object_slug:
            query += " WHERE json_extract(record,'$.object_slug')=?"
            args = (object_slug,)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [RunRecord.model_validate_json(row["record"]) for row in rows]

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
        *,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> EventRecord:
        event_id = event_id or str(uuid.uuid4())
        created_at = created_at or datetime.now(UTC).isoformat()
        with self._db() as db:
            duplicate = db.execute(
                "SELECT sequence,event_type,payload,created_at FROM run_events WHERE run_id=? AND event_id=?",
                (run_id, event_id),
            ).fetchone()
            if duplicate:
                return EventRecord(
                    run_id,
                    int(duplicate["sequence"]),
                    event_id,
                    duplicate["event_type"],
                    json.loads(duplicate["payload"]),
                    duplicate["created_at"],
                )
            if not db.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
                raise KeyError(f"run not found: {run_id}")
            sequence = int(
                db.execute(
                    "SELECT COALESCE(MAX(sequence),0)+1 FROM run_events WHERE run_id=?", (run_id,)
                ).fetchone()[0]
            )
            db.execute(
                "INSERT INTO run_events VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    sequence,
                    event_id,
                    event_type,
                    json.dumps(payload, sort_keys=True),
                    created_at,
                ),
            )
        return EventRecord(run_id, sequence, event_id, event_type, payload, created_at)

    def read_events(
        self, run_id: str, *, cursor: int = 0, limit: int = 100
    ) -> tuple[list[EventRecord], int | None]:
        if cursor < 0 or limit < 1 or limit > 1000:
            raise ValueError("cursor must be non-negative and limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM run_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (run_id, cursor, limit + 1),
            ).fetchall()
        events = [
            EventRecord(
                run_id,
                int(row["sequence"]),
                row["event_id"],
                row["event_type"],
                json.loads(row["payload"]),
                row["created_at"],
            )
            for row in rows[:limit]
        ]
        next_cursor = events[-1].sequence if len(rows) > limit and events else None
        return events, next_cursor

    def compact(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT projection,record FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            cursor = db.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM run_events WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        if not row:
            return None
        result = (
            json.loads(row["projection"])
            if row["projection"]
            else compact_status(RunRecord.model_validate_json(row["record"]))
        )
        result["event_cursor"] = int(cursor)
        return result


class PostgresRunStore:
    """Cloud SQL PostgreSQL implementation. Supports a psycopg URL or injected connector."""

    def __init__(self, database_url: str | None = None, connect: Callable[..., Any] | None = None):
        self.database_url = database_url or os.environ.get("DEFECT_STATE_DATABASE_URL")
        self._connect_fn = connect
        if not self.database_url and connect is None:
            raise ValueError("DEFECT_STATE_DATABASE_URL or a connection factory is required")
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS defect_runs (
                run_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                fingerprint TEXT NOT NULL, payload JSONB NOT NULL, record JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL, projection JSONB)""")
            db.execute("ALTER TABLE defect_runs ADD COLUMN IF NOT EXISTS projection JSONB")
            db.execute("""CREATE TABLE IF NOT EXISTS defect_run_events (
                run_id TEXT NOT NULL, sequence BIGINT NOT NULL,
                event_id TEXT NOT NULL, event_type TEXT NOT NULL,
                payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(run_id, sequence), UNIQUE(run_id, event_id))""")

    def _connect(self):
        if self._connect_fn:
            return self._connect_fn(self.database_url)
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL run storage requires the optional psycopg dependency"
            ) from exc
        database_url = self.database_url
        if database_url is None:
            raise RuntimeError("PostgreSQL connection URL is not configured")
        return psycopg.connect(database_url)

    @contextmanager
    def _db(self):
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _load(value):
        return json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value

    def create(
        self, run: RunRecord, request_fingerprint: str, payload: dict
    ) -> tuple[RunRecord, bool]:
        key = str(payload["idempotency_key"])
        with self._db() as db:
            inserted = db.execute(
                """INSERT INTO defect_runs
                (run_id,idempotency_key,fingerprint,payload,record,created_at,projection)
                VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb)
                ON CONFLICT (idempotency_key) DO NOTHING RETURNING record""",
                (
                    run.run_id,
                    key,
                    request_fingerprint,
                    json.dumps(payload),
                    run.model_dump_json(),
                    run.created_at,
                    json.dumps(compact_status(run)),
                ),
            ).fetchone()
            if inserted:
                db.execute(
                    "INSERT INTO defect_run_events VALUES (%s,%s,%s,%s,%s::jsonb,%s)",
                    (
                        run.run_id,
                        1,
                        f"{run.run_id}:run.created",
                        "run.created",
                        json.dumps({"state": run.state.value}),
                        run.created_at,
                    ),
                )
                return RunRecord.model_validate(self._load(inserted[0])), True
            row = db.execute(
                "SELECT fingerprint,record FROM defect_runs WHERE idempotency_key=%s", (key,)
            ).fetchone()
            if row is None:
                raise RuntimeError("idempotency conflict occurred without a visible run record")
            if row[0] != request_fingerprint:
                raise ValueError("idempotency key was already used for a different request")
            return RunRecord.model_validate(self._load(row[1])), False

    def get(self, run_id: str) -> RunRecord | None:
        with self._db() as db:
            row = db.execute("SELECT record FROM defect_runs WHERE run_id=%s", (run_id,)).fetchone()
        return RunRecord.model_validate(self._load(row[0])) if row else None

    def by_key(self, key: str) -> RunRecord | None:
        with self._db() as db:
            row = db.execute(
                "SELECT record FROM defect_runs WHERE idempotency_key=%s", (key,)
            ).fetchone()
        return RunRecord.model_validate(self._load(row[0])) if row else None

    def get_payload(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute(
                "SELECT payload FROM defect_runs WHERE run_id=%s", (run_id,)
            ).fetchone()
        return self._load(row[0]) if row else None

    def update(self, run: RunRecord) -> None:
        with self._db() as db:
            result = db.execute(
                "UPDATE defect_runs SET record=%s::jsonb,projection=%s::jsonb WHERE run_id=%s",
                (run.model_dump_json(), json.dumps(compact_status(run)), run.run_id),
            )
            if result.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")

    def update_with_event(
        self,
        run: RunRecord,
        event_type: str,
        payload: dict[str, Any],
        *,
        event_id: str | None = None,
    ) -> None:
        event_id = event_id or str(uuid.uuid4())
        with self._db() as db:
            result = db.execute(
                "UPDATE defect_runs SET record=%s::jsonb,projection=%s::jsonb WHERE run_id=%s",
                (run.model_dump_json(), json.dumps(compact_status(run)), run.run_id),
            )
            if result.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")
            db.execute("SELECT run_id FROM defect_runs WHERE run_id=%s FOR UPDATE", (run.run_id,))
            sequence_row = db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM defect_run_events WHERE run_id=%s",
                (run.run_id,),
            ).fetchone()
            if sequence_row is None:
                raise RuntimeError("could not allocate a control event sequence")
            sequence = int(sequence_row[0])
            db.execute(
                "INSERT INTO defect_run_events VALUES (%s,%s,%s,%s,%s::jsonb,%s)",
                (
                    run.run_id,
                    sequence,
                    event_id,
                    event_type,
                    json.dumps(payload),
                    datetime.now(UTC),
                ),
            )

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        if object_slug:
            query, args = (
                "SELECT record FROM defect_runs WHERE record->>'object_slug'=%s ORDER BY created_at DESC",
                (object_slug,),
            )
        else:
            query, args = "SELECT record FROM defect_runs ORDER BY created_at DESC", ()
        with self._db() as db:
            rows = db.execute(query, args).fetchall()
        return [RunRecord.model_validate(self._load(row[0])) for row in rows]

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
        *,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> EventRecord:
        event_id = event_id or str(uuid.uuid4())
        created_at = created_at or datetime.now(UTC).isoformat()
        with self._db() as db:
            duplicate = db.execute(
                "SELECT sequence,event_type,payload,created_at FROM defect_run_events WHERE run_id=%s AND event_id=%s",
                (run_id, event_id),
            ).fetchone()
            if duplicate:
                return EventRecord(
                    run_id,
                    int(duplicate[0]),
                    event_id,
                    duplicate[1],
                    self._load(duplicate[2]),
                    str(duplicate[3]),
                )
            if (
                db.execute("SELECT 1 FROM defect_runs WHERE run_id=%s", (run_id,)).fetchone()
                is None
            ):
                raise KeyError(f"run not found: {run_id}")
            # Serialize allocation per run so concurrent callbacks cannot share a cursor.
            db.execute("SELECT run_id FROM defect_runs WHERE run_id=%s FOR UPDATE", (run_id,))
            row = db.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM defect_run_events WHERE run_id=%s",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("could not allocate a control event sequence")
            sequence = int(row[0])
            db.execute(
                "INSERT INTO defect_run_events VALUES (%s,%s,%s,%s,%s::jsonb,%s)",
                (run_id, sequence, event_id, event_type, json.dumps(payload), created_at),
            )
        return EventRecord(run_id, sequence, event_id, event_type, payload, created_at)

    def read_events(
        self, run_id: str, *, cursor: int = 0, limit: int = 100
    ) -> tuple[list[EventRecord], int | None]:
        if cursor < 0 or limit < 1 or limit > 1000:
            raise ValueError("cursor must be non-negative and limit must be between 1 and 1000")
        with self._db() as db:
            rows = db.execute(
                "SELECT sequence,event_id,event_type,payload,created_at FROM defect_run_events WHERE run_id=%s AND sequence>%s ORDER BY sequence LIMIT %s",
                (run_id, cursor, limit + 1),
            ).fetchall()
        events = [
            EventRecord(run_id, int(row[0]), row[1], row[2], self._load(row[3]), str(row[4]))
            for row in rows[:limit]
        ]
        return events, (events[-1].sequence if len(rows) > limit and events else None)

    def compact(self, run_id: str) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute(
                "SELECT projection,record FROM defect_runs WHERE run_id=%s", (run_id,)
            ).fetchone()
            cursor_row = db.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM defect_run_events WHERE run_id=%s", (run_id,)
            ).fetchone()
            if cursor_row is None:
                raise RuntimeError("could not read the control event cursor")
            cursor = cursor_row[0]
        if not row:
            return None
        result = (
            self._load(row[0])
            if row[0]
            else compact_status(RunRecord.model_validate(self._load(row[1])))
        )
        result["event_cursor"] = int(cursor)
        return result


class PostgresIntakeSessionStore:
    """PostgreSQL implementation for Cloud SQL deployments."""

    def __init__(self, database_url: str | None = None, connect: Callable[..., Any] | None = None):
        self.database_url = database_url or os.environ.get("DEFECT_STATE_DATABASE_URL")
        self._connect_fn = connect
        if not self.database_url and connect is None:
            raise ValueError("DEFECT_STATE_DATABASE_URL or a connection factory is required")
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS defect_intake_sessions (
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                schema_ref TEXT NOT NULL, draft JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL)""")

    def _connect(self):
        if self._connect_fn:
            return self._connect_fn(self.database_url)
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL intake storage requires the optional psycopg dependency"
            ) from exc
        database_url = self.database_url
        if database_url is None:
            raise RuntimeError("PostgreSQL connection URL is not configured")
        return psycopg.connect(database_url)

    @contextmanager
    def _db(self):
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _load(value):
        return json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value

    def _decode(self, row) -> IntakeSession:
        return IntakeSession.model_validate(
            {
                "session_id": row[0],
                "revision": row[1],
                "schema_ref": row[2],
                "draft": self._load(row[3]),
                "created_at": row[4],
                "updated_at": row[5],
                "expires_at": row[6],
            }
        )

    def create(self, session: IntakeSession) -> IntakeSession:
        with self._db() as db:
            try:
                db.execute(
                    """INSERT INTO defect_intake_sessions
                    (session_id,revision,schema_ref,draft,created_at,updated_at,expires_at)
                    VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s)""",
                    (
                        session.session_id,
                        session.revision,
                        session.schema_ref,
                        session.draft.model_dump_json(),
                        session.created_at,
                        session.updated_at,
                        session.expires_at,
                    ),
                )
            except Exception as exc:
                if getattr(exc, "sqlstate", None) == "23505":
                    raise ValueError(
                        f"intake session already exists: {session.session_id}"
                    ) from exc
                raise
        return session

    def get(self, session_id: str, *, now: datetime | None = None) -> IntakeSession | None:
        with self._db() as db:
            row = db.execute(
                """SELECT session_id,revision,schema_ref,draft,created_at,updated_at,expires_at
                FROM defect_intake_sessions WHERE session_id=%s AND expires_at>%s""",
                (session_id, _session_now(now)),
            ).fetchone()
        return self._decode(row) if row else None

    def update(self, session: IntakeSession, expected_revision: int) -> IntakeSession:
        now = _session_now()
        if session.revision != expected_revision + 1:
            raise IntakeSessionConflictError(
                f"intake session revision conflict: expected {expected_revision}, submitted {session.revision}"
            )
        with self._db() as db:
            result = db.execute(
                """UPDATE defect_intake_sessions SET revision=%s,schema_ref=%s,
                draft=%s::jsonb,updated_at=%s,expires_at=%s
                WHERE session_id=%s AND revision=%s AND expires_at>%s""",
                (
                    session.revision,
                    session.schema_ref,
                    session.draft.model_dump_json(),
                    session.updated_at,
                    session.expires_at,
                    session.session_id,
                    expected_revision,
                    now,
                ),
            ).rowcount
            if result == 1:
                return session
            row = db.execute(
                """SELECT session_id,revision,schema_ref,draft,created_at,updated_at,expires_at
                FROM defect_intake_sessions WHERE session_id=%s""",
                (session.session_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"intake session not found: {session.session_id}")
        current = self._decode(row)
        if current.expires_at <= now:
            raise IntakeSessionExpiredError(session.session_id)
        raise IntakeSessionConflictError(
            f"intake session revision conflict: expected {expected_revision}, current {current.revision}",
            current,
        )

    def delete_expired(self, *, now: datetime | None = None) -> int:
        with self._db() as db:
            return db.execute(
                "DELETE FROM defect_intake_sessions WHERE expires_at<=%s", (_session_now(now),)
            ).rowcount


def intake_store_from_env() -> IntakeSessionStore:
    """Select the session store using the same database setting as runs."""
    url = database_url_from_env()
    if url.startswith("sqlite://"):
        return SQLiteIntakeSessionStore(url.removeprefix("sqlite://"))
    if url.startswith(("postgres://", "postgresql://")):
        return PostgresIntakeSessionStore(url)
    raise ValueError("DEFECT_STATE_DATABASE_URL must use sqlite:/// or postgresql://")


def database_url_from_env() -> str:
    """Resolve either a direct URL or the Cloud Run Cloud SQL Unix socket settings."""
    direct = os.environ.get("DEFECT_STATE_DATABASE_URL")
    if direct:
        return direct
    database = os.environ.get("DEFECT_STATE_DATABASE")
    user = os.environ.get("DEFECT_STATE_DB_USER")
    socket = os.environ.get("DEFECT_STATE_DB_SOCKET")
    if database and user and socket:
        password = os.environ.get("DEFECT_STATE_DB_PASSWORD", "")
        auth = quote_plus(user) + (":" + quote_plus(password) if password else "")
        return f"postgresql://{auth}@/{quote_plus(database)}?host={quote_plus(socket)}"
    return "sqlite://./.defect/runs.sqlite3"


def run_store_from_env() -> RunStore:
    """Select the durable store from DEFECT_STATE_DATABASE_URL."""
    url = database_url_from_env()
    if url.startswith("sqlite://"):
        return SQLiteRunStore(url.removeprefix("sqlite://"))
    if url.startswith(("postgres://", "postgresql://")):
        return PostgresRunStore(url)
    raise ValueError("DEFECT_STATE_DATABASE_URL must use sqlite:/// or postgresql://")
