"""Durable release history for local SQLite and Cloud SQL PostgreSQL."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from defect_platform.contracts import ModelRelease


class SqlReleaseLedger:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.postgres = database_url.startswith(("postgres://", "postgresql://"))
        if not self.postgres:
            if database_url.startswith("sqlite://"):
                path = database_url.removeprefix("sqlite://")
            else:
                path = database_url
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.database_url = path
        self._initialize()

    def _connect(self):
        if self.postgres:
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError("PostgreSQL release ledger requires psycopg") from exc
            return psycopg.connect(self.database_url)
        return sqlite3.connect(self.database_url)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_releases ("
                "release_id TEXT PRIMARY KEY, object_slug TEXT NOT NULL, "
                "state TEXT NOT NULL, approved_at TEXT, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS model_releases_by_object "
                "ON model_releases(object_slug, approved_at)"
            )

    def save(self, release: ModelRelease) -> None:
        values = (
            release.release_id, release.object_slug, release.state,
            release.approved_at.isoformat() if release.approved_at else None,
            release.model_dump_json(),
        )
        with self._connect() as connection:
            if self.postgres:
                connection.execute(
                    "INSERT INTO model_releases VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT(release_id) DO UPDATE SET state=excluded.state, "
                    "approved_at=excluded.approved_at, payload=excluded.payload", values,
                )
            else:
                connection.execute(
                    "INSERT INTO model_releases VALUES (?,?,?,?,?) "
                    "ON CONFLICT(release_id) DO UPDATE SET state=excluded.state, "
                    "approved_at=excluded.approved_at, payload=excluded.payload", values,
                )

    def get(self, release_id: str) -> ModelRelease:
        placeholder = "%s" if self.postgres else "?"
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT payload FROM model_releases WHERE release_id={placeholder}", (release_id,)
            ).fetchone()
        if row is None:
            raise KeyError(release_id)
        return ModelRelease.model_validate_json(row[0])

    def _promoted(self, object_slug: str) -> list[ModelRelease]:
        placeholder = "%s" if self.postgres else "?"
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM model_releases WHERE object_slug=" + placeholder
                + " AND state='promoted' ORDER BY approved_at DESC", (object_slug,)
            ).fetchall()
        return [ModelRelease.model_validate_json(row[0]) for row in rows]

    def latest_promoted(self, object_slug: str) -> ModelRelease | None:
        releases = self._promoted(object_slug)
        return releases[0] if releases else None

    def previous_promoted(self, object_slug: str, release_id: str) -> ModelRelease | None:
        releases = self._promoted(object_slug)
        for index, release in enumerate(releases):
            if release.release_id == release_id:
                return releases[index + 1] if index + 1 < len(releases) else None
        return None


class GCSReleaseLedger:
    """Append-only release history usable by an operator outside Cloud SQL."""

    def __init__(self, root_uri: str, *, client=None):
        parsed = urlparse(root_uri)
        if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("release ledger URI must be a gs://bucket/prefix")
        self.bucket_name = parsed.netloc
        self.prefix = parsed.path.strip("/")
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError("GCS release ledger requires cloud dependencies") from exc
            client = storage.Client()
        self.client = client

    @staticmethod
    def _release_id(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value):
            raise ValueError("release_id contains invalid GCS path characters")
        return value

    def save(self, release: ModelRelease) -> None:
        release_id = self._release_id(release.release_id)
        name = (f"{self.prefix}/{release_id}/"
                f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid4().hex}.json")
        self.client.bucket(self.bucket_name).blob(name).upload_from_string(
            release.model_dump_json(), content_type="application/json", if_generation_match=0
        )

    def get(self, release_id: str) -> ModelRelease:
        release_id = self._release_id(release_id)
        prefix = f"{self.prefix}/{release_id}/"
        revisions = list(self.client.list_blobs(self.bucket_name, prefix=prefix))
        if not revisions:
            raise KeyError(release_id)
        return ModelRelease.model_validate_json(max(revisions, key=lambda item: item.name).download_as_bytes())

    def _promoted(self, object_slug: str) -> list[ModelRelease]:
        latest: dict[str, object] = {}
        prefix = self.prefix + "/"
        for blob in self.client.list_blobs(self.bucket_name, prefix=prefix):
            parts = blob.name[len(prefix):].split("/", 1)
            if len(parts) != 2:
                continue
            release_id = parts[0]
            if release_id not in latest or blob.name > latest[release_id].name:
                latest[release_id] = blob
        releases = [ModelRelease.model_validate_json(blob.download_as_bytes()) for blob in latest.values()]
        promoted = [item for item in releases if item.object_slug == object_slug and item.state == "promoted"]
        return sorted(promoted, key=lambda item: item.approved_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    def latest_promoted(self, object_slug: str) -> ModelRelease | None:
        releases = self._promoted(object_slug)
        return releases[0] if releases else None

    def previous_promoted(self, object_slug: str, release_id: str) -> ModelRelease | None:
        releases = self._promoted(object_slug)
        for index, release in enumerate(releases):
            if release.release_id == release_id:
                return releases[index + 1] if index + 1 < len(releases) else None
        return None
