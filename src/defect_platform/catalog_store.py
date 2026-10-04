"""Immutable approved class catalogs, resolved from an operator configured root.

Checksums detect substitution relative to a pinned digest. The store's filesystem
permissions or GCS IAM establish approval authority; reviewer text is audit data.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

from defect_platform.semantics import SHA256, ClassCatalog, canonical_bytes, fingerprint


class ClassCatalogStore(Protocol):
    def get_catalog(self, object_slug: str, catalog_sha256: str) -> ClassCatalog | None: ...


T = TypeVar("T", bound=BaseModel)


class ReferenceCatalogStore(Protocol):
    """Immutable operator-owned records addressed by kind, ID, version, and digest."""

    def get_reference(
        self, kind: str, record_id: str, version: str | None, sha256: str, model: type[T]
    ) -> T | None: ...

    def publish_reference(
        self, kind: str, record_id: str, version: str, value: Any, sha256: str | None = None
    ) -> str: ...


def serialize_catalog(catalog: ClassCatalog) -> bytes:
    catalog = ClassCatalog.model_validate(catalog.model_dump(mode="json"))
    if catalog.review_status != "reviewed":
        raise ValueError("only reviewed class catalogs can enter the approval store")
    return canonical_bytes({"catalog": catalog.model_dump(mode="json"), "sha256": catalog.sha256})


def parse_catalog(payload: bytes, expected_sha256: str) -> ClassCatalog:
    envelope = json.loads(payload)
    if not isinstance(envelope, dict) or set(envelope) != {"catalog", "sha256"}:
        raise ValueError("invalid approved catalog envelope")
    catalog = ClassCatalog.model_validate(envelope["catalog"])
    if catalog.review_status != "reviewed":
        raise ValueError("catalog is not reviewed")
    if envelope["sha256"] != catalog.sha256 or catalog.sha256 != expected_sha256:
        raise ValueError("approved catalog fingerprint mismatch")
    return catalog


class DirectoryCatalogStore:
    """One create-only JSON envelope per object and catalog fingerprint.

    Supports local directories and gs:// roots. Admission gets read credentials;
    catalog publication uses separate operator credentials.
    """

    def __init__(self, root: str | Path, *, client=None):
        self.root = str(root).rstrip("/")
        if not self.root:
            raise ValueError("catalog store root is required")
        self.client = client
        if self.root.startswith("gs://"):
            parsed = urlparse(self.root)
            if not parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError("invalid GCS catalog root")
        elif "://" in self.root:
            raise ValueError("catalog stores support local directories or GCS")

    def _key(self, object_slug: str, digest: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", object_slug) or not re.fullmatch(SHA256, digest):
            raise ValueError("invalid catalog object or fingerprint")
        return f"{object_slug}/{digest}.json"

    def _path(self, key: str) -> Path:
        root = Path(self.root).resolve()
        path = root / key
        if not path.resolve().is_relative_to(root):
            raise ValueError("catalog path escapes its configured root")
        return path

    def _blob(self, key: str):
        if self.client is None:
            from google.cloud import storage

            self.client = storage.Client()
        parsed = urlparse(self.root)
        name = "/".join(part for part in (parsed.path.strip("/"), key) if part)
        return self.client.bucket(parsed.netloc).blob(name)

    def get_catalog(self, object_slug: str, catalog_sha256: str) -> ClassCatalog | None:
        key = self._key(object_slug, catalog_sha256)
        if self.root.startswith("gs://"):
            blob = self._blob(key)
            try:
                payload = blob.download_as_bytes()
            except Exception as exc:
                if getattr(exc, "code", None) == 404:
                    return None
                raise
        else:
            try:
                payload = self._path(key).read_bytes()
            except FileNotFoundError:
                return None
        catalog = parse_catalog(payload, catalog_sha256)
        if catalog.object_slug != object_slug:
            raise ValueError("approved catalog object mismatch")
        return catalog

    def publish(self, catalog: ClassCatalog) -> str:
        payload = serialize_catalog(catalog)
        key = self._key(catalog.object_slug, catalog.sha256)
        # A version number cannot be reused for changed meaning or approval.
        version_key = (
            f"{catalog.object_slug}/versions/"
            f"{fingerprint(catalog.catalog_id)}-{catalog.version}.json"
        )
        self._create(version_key, canonical_bytes({"sha256": catalog.sha256}))
        self._create(key, payload)
        return f"{self.root}/{key}" if self.root.startswith("gs://") else str(self._path(key))

    def _create(self, key: str, payload: bytes) -> None:
        if self.root.startswith("gs://"):
            blob = self._blob(key)
            try:
                blob.upload_from_string(
                    payload, content_type="application/json", if_generation_match=0
                )
            except Exception as exc:
                if getattr(exc, "code", None) not in (409, 412):
                    raise
                if blob.download_as_bytes() != payload:
                    raise ValueError("immutable catalog already contains different bytes") from exc
            return
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError("immutable catalog already contains different bytes")


class DirectoryReferenceCatalogStore:
    """Create-only typed records sharing the operator catalog root.

    Architecture, preprocessing, loss, evaluator, deployment, snippet, preset,
    and operator-profile records all use this same authority boundary. The
    record body is content-addressed and never overwritten.
    """

    def __init__(self, root: str | Path):
        self.root = str(root).rstrip("/")
        if not self.root or self.root.startswith("gs://"):
            raise ValueError("reference catalogs currently require a local operator directory")

    def _path(self, kind: str, record_id: str, version: str, sha256: str) -> Path:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", kind):
            raise ValueError("invalid reference kind")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", record_id) or not re.fullmatch(
            r"[A-Za-z0-9._-]+", version
        ):
            raise ValueError("invalid reference identity")
        if not re.fullmatch(SHA256, sha256):
            raise ValueError("invalid reference checksum")
        sha256 = sha256.lower()
        root = Path(self.root).resolve()
        path = root / "references" / kind / record_id / f"{version}-{sha256}.json"
        if not path.resolve().is_relative_to(root):
            raise ValueError("reference path escapes configured root")
        return path

    def publish_reference(
        self, kind: str, record_id: str, version: str, value: Any, sha256: str | None = None
    ) -> str:
        body = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        actual_digest = fingerprint(body)
        if sha256 is not None and sha256.lower() != actual_digest:
            raise ValueError("supplied reference checksum does not match record content")
        digest = actual_digest
        path = self._path(kind, record_id, version, digest)
        payload = canonical_bytes(
            {"kind": kind, "id": record_id, "version": version, "sha256": digest, "value": body}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("immutable reference already contains different bytes")
        return str(path)

    def get_reference(
        self, kind: str, record_id: str, version: str | None, sha256: str, model: type[T]
    ) -> T | None:
        sha256 = sha256.lower()
        directory = Path(self.root) / "references" / kind / record_id
        candidates = sorted(directory.glob("*.json"))
        if version is not None:
            candidates = [self._path(kind, record_id, version, sha256), *candidates]
        for path in candidates:
            try:
                envelope = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            if envelope.get("kind") != kind or envelope.get("id") != record_id:
                continue
            if version is not None and envelope.get("version") != version:
                continue
            value = model.model_validate(envelope["value"])
            if envelope.get("sha256") == sha256 or getattr(value, "sha256", None) == sha256:
                if fingerprint(envelope.get("value")) != envelope.get("sha256"):
                    raise ValueError("reference checksum mismatch")
                return value
        return None


class GCSReferenceCatalogStore:
    """Immutable reference records stored under an operator GCS prefix."""

    def __init__(self, root: str, *, client=None):
        parsed = urlparse(root.rstrip("/"))
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError("GCS reference catalog root must use gs://bucket/prefix")
        self.root = root.rstrip("/")
        self.bucket_name, self.prefix = parsed.netloc, parsed.path.lstrip("/")
        if self.prefix:
            self.prefix += "/"
        self._client = client

    @property
    def bucket(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError(
                    "GCS reference catalogs require the cloud dependency group"
                ) from exc
            self._client = storage.Client()
        return self._client.bucket(self.bucket_name)

    @staticmethod
    def _name(kind: str, record_id: str, version: str, sha256: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", kind):
            raise ValueError("invalid reference kind")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", record_id) or not re.fullmatch(
            r"[A-Za-z0-9._-]+", version
        ):
            raise ValueError("invalid reference identity")
        if not re.fullmatch(SHA256, sha256):
            raise ValueError("invalid reference checksum")
        return f"references/{kind}/{record_id}/{version}-{sha256.lower()}.json"

    def publish_reference(
        self, kind: str, record_id: str, version: str, value: Any, sha256: str | None = None
    ) -> str:
        body = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        digest = fingerprint(body)
        if sha256 is not None and sha256.lower() != digest:
            raise ValueError("supplied reference checksum does not match record content")
        payload = canonical_bytes(
            {"kind": kind, "id": record_id, "version": version, "sha256": digest, "value": body}
        )
        blob = self.bucket.blob(self.prefix + self._name(kind, record_id, version, digest))
        try:
            blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
        except Exception as exc:
            if getattr(exc, "code", None) not in (409, 412):
                raise
            if blob.download_as_bytes() != payload:
                raise ValueError("immutable reference already contains different bytes") from exc
        return f"{self.root}/{self._name(kind, record_id, version, digest)}"

    def get_reference(
        self, kind: str, record_id: str, version: str | None, sha256: str, model: type[T]
    ) -> T | None:
        digest = sha256.lower()
        candidates = list(
            self.bucket.list_blobs(prefix=self.prefix + f"references/{kind}/{record_id}/")
        )
        if version is not None:
            exact = self.bucket.blob(self.prefix + self._name(kind, record_id, version, digest))
            candidates = [exact, *candidates]
        for blob in candidates:
            try:
                envelope = json.loads(blob.download_as_bytes())
            except Exception as exc:
                if getattr(exc, "code", None) == 404:
                    continue
                raise
            if envelope.get("kind") != kind or envelope.get("id") != record_id:
                continue
            if version is not None and envelope.get("version") != version:
                continue
            value = model.model_validate(envelope["value"])
            if envelope.get("sha256") == digest or getattr(value, "sha256", None) == digest:
                if fingerprint(envelope.get("value")) != envelope.get("sha256"):
                    raise ValueError("reference checksum mismatch")
                return value
        return None


def catalog_store_from_env() -> DirectoryCatalogStore | None:
    root = os.getenv("DEFECT_CLASS_CATALOG_ROOT")
    return DirectoryCatalogStore(root) if root else None


def reference_catalog_store_from_env() -> ReferenceCatalogStore | None:
    root = os.getenv("DEFECT_REFERENCE_CATALOG_ROOT")
    if not root:
        class_root = os.getenv("DEFECT_CLASS_CATALOG_ROOT")
        root = class_root if class_root else None
    if not root:
        return None
    if root.startswith("gs://"):
        return GCSReferenceCatalogStore(root)
    return DirectoryReferenceCatalogStore(root)
