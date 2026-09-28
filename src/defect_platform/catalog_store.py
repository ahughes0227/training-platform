"""Immutable approved class catalogs, resolved from an operator configured root.

Checksums detect substitution relative to a pinned digest. The store's filesystem
permissions or GCS IAM establish approval authority; reviewer text is audit data.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from defect_platform.semantics import SHA256, ClassCatalog, canonical_bytes, fingerprint


class ClassCatalogStore(Protocol):
    def get_catalog(self, object_slug: str, catalog_sha256: str) -> ClassCatalog | None: ...


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
        version_key = (f"{catalog.object_slug}/versions/"
                       f"{fingerprint(catalog.catalog_id)}-{catalog.version}.json")
        self._create(version_key, canonical_bytes({"sha256": catalog.sha256}))
        self._create(key, payload)
        return f"{self.root}/{key}" if self.root.startswith("gs://") else str(self._path(key))

    def _create(self, key: str, payload: bytes) -> None:
        if self.root.startswith("gs://"):
            blob = self._blob(key)
            try:
                blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
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


def catalog_store_from_env() -> DirectoryCatalogStore | None:
    root = os.getenv("DEFECT_CLASS_CATALOG_ROOT")
    return DirectoryCatalogStore(root) if root else None
