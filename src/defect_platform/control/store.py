"""Durable control-plane state. SQLite is useful locally; production stores are injectable."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Protocol, Any
from urllib.parse import quote_plus

from defect_platform.contracts import RunRecord


class RunStore(Protocol):
    def create(self, run: RunRecord, request_fingerprint: str, payload: dict) -> tuple[RunRecord, bool]: ...
    def get(self, run_id: str) -> RunRecord | None: ...
    def by_key(self, key: str) -> RunRecord | None: ...
    def get_payload(self, run_id: str) -> dict | None: ...
    def update(self, run: RunRecord) -> None: ...
    def list(self, object_slug: str | None = None) -> list[RunRecord]: ...


class SQLiteRunStore:
    """SQLite-backed store with unique idempotency keys and atomic record updates."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                fingerprint TEXT NOT NULL, payload TEXT NOT NULL, record TEXT NOT NULL,
                created_at TEXT NOT NULL)""")

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

    def create(self, run: RunRecord, request_fingerprint: str, payload: dict) -> tuple[RunRecord, bool]:
        key = str(payload["idempotency_key"])
        with self._db() as db:
            row = db.execute("SELECT fingerprint,record FROM runs WHERE idempotency_key=?", (key,)).fetchone()
            if row:
                if row["fingerprint"] != request_fingerprint:
                    raise ValueError("idempotency key was already used for a different request")
                return RunRecord.model_validate_json(row["record"]), False
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)", (
                run.run_id, key, request_fingerprint, json.dumps(payload, sort_keys=True),
                run.model_dump_json(), run.created_at.isoformat()))
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
            result = db.execute("UPDATE runs SET record=? WHERE run_id=?", (run.model_dump_json(), run.run_id))
            if result.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        query, args = "SELECT record FROM runs", ()
        if object_slug:
            query += " WHERE json_extract(record,'$.object_slug')=?"
            args = (object_slug,)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [RunRecord.model_validate_json(row["record"]) for row in rows]


class PostgresRunStore:
    """Cloud SQL PostgreSQL implementation. Supports a psycopg URL or injected connector."""

    def __init__(self, database_url: str | None = None,
                 connect: Callable[..., Any] | None = None):
        self.database_url = database_url or os.environ.get("DEFECT_STATE_DATABASE_URL")
        self._connect_fn = connect
        if not self.database_url and connect is None:
            raise ValueError("DEFECT_STATE_DATABASE_URL or a connection factory is required")
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS defect_runs (
                run_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                fingerprint TEXT NOT NULL, payload JSONB NOT NULL, record JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL)""")

    def _connect(self):
        if self._connect_fn:
            return self._connect_fn(self.database_url)
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL run storage requires the optional psycopg dependency") from exc
        return psycopg.connect(self.database_url)

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

    def create(self, run: RunRecord, request_fingerprint: str, payload: dict) -> tuple[RunRecord, bool]:
        key = str(payload["idempotency_key"])
        with self._db() as db:
            inserted = db.execute("""INSERT INTO defect_runs
                (run_id,idempotency_key,fingerprint,payload,record,created_at)
                VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                ON CONFLICT (idempotency_key) DO NOTHING RETURNING record""",
                (run.run_id, key, request_fingerprint, json.dumps(payload),
                 run.model_dump_json(), run.created_at)).fetchone()
            if inserted:
                return RunRecord.model_validate(self._load(inserted[0])), True
            row = db.execute("SELECT fingerprint,record FROM defect_runs WHERE idempotency_key=%s",
                             (key,)).fetchone()
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
            row = db.execute("SELECT record FROM defect_runs WHERE idempotency_key=%s", (key,)).fetchone()
        return RunRecord.model_validate(self._load(row[0])) if row else None

    def get_payload(self, run_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT payload FROM defect_runs WHERE run_id=%s", (run_id,)).fetchone()
        return self._load(row[0]) if row else None

    def update(self, run: RunRecord) -> None:
        with self._db() as db:
            result = db.execute("UPDATE defect_runs SET record=%s::jsonb WHERE run_id=%s",
                                (run.model_dump_json(), run.run_id))
            if result.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        if object_slug:
            query, args = "SELECT record FROM defect_runs WHERE record->>'object_slug'=%s ORDER BY created_at DESC", (object_slug,)
        else:
            query, args = "SELECT record FROM defect_runs ORDER BY created_at DESC", ()
        with self._db() as db:
            rows = db.execute(query, args).fetchall()
        return [RunRecord.model_validate(self._load(row[0])) for row in rows]


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
