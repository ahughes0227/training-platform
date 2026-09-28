"""Observed deployment output consumed by the training control plane.

This is an operator-pinned acceptance record, not a provider probe. Live tests
must produce its evidence and IAM must protect its configured location.
"""
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from defect_platform.semantics import SHA256, SemanticRecord, canonical_bytes, fingerprint


class InfrastructureCapabilities(SemanticRecord):
    schema_version: Literal[1] = 1
    capability_id: str = Field(min_length=1)
    deployment_revision: str = Field(pattern=SHA256)
    project: str = Field(min_length=1)
    region: str = Field(min_length=1)
    observed_at: datetime
    expires_at: datetime
    readiness: Literal["ready", "unready"]
    dataset_root_uri: str
    artifact_root_uri: str
    trainer_service_account: str
    control_url: str
    mlflow_uri: str
    otlp_endpoint: str | None = None
    serving_url: str | None = None
    certified_sources: dict[str, str] = Field(min_length=1)  # exact image digest -> source commit
    supported_contracts: dict[str, int] = Field(default_factory=lambda: {
        "class_catalog": 1, "dataset_semantics": 1, "model_semantics": 1})
    evidence_uris: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self):
        for value in (self.observed_at, self.expires_at):
            if value.tzinfo is None:
                raise ValueError("capability observation times must include timezone")
        if self.expires_at <= self.observed_at:
            raise ValueError("capability expiry must follow observation")
        for root in (self.dataset_root_uri, self.artifact_root_uri):
            parsed = urlparse(root)
            if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError("infrastructure artifact roots must be GCS URIs")
        for endpoint in (self.control_url, self.mlflow_uri, self.otlp_endpoint, self.serving_url):
            if endpoint and (urlparse(endpoint).scheme != "https" or not urlparse(endpoint).netloc):
                raise ValueError("observed production endpoints must use HTTPS")
        if any(not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", digest)
               or not re.fullmatch(r"[a-f0-9]{40}", commit)
               for digest, commit in self.certified_sources.items()):
            raise ValueError("capability runtimes require exact image digests and source commits")
        if self.supported_contracts != {
                "class_catalog": 1, "dataset_semantics": 1, "model_semantics": 1}:
            raise ValueError("unsupported module contract versions")
        if not all(value.strip() for value in self.evidence_uris):
            raise ValueError("capability evidence locations cannot be blank")
        return self

    @property
    def sha256(self) -> str:
        return fingerprint(self)

    def validate_training(self, job, runtime, dataset, *, now: datetime | None = None) -> None:
        # Revalidate nested mutations before accepting a previously parsed object.
        value = InfrastructureCapabilities.model_validate(self.model_dump(mode="json"))
        current = now or datetime.now(UTC)
        if value.readiness != "ready" or not value.observed_at <= current < value.expires_at:
            raise ValueError("infrastructure capability is unready, stale, or not yet observed")
        if (job.project, job.region, job.service_account) != (
                value.project, value.region, value.trainer_service_account):
            raise ValueError("job identity differs from observed infrastructure")
        if value.certified_sources.get(runtime.image_digest) != runtime.source_commit:
            raise ValueError("runtime revision is absent from observed infrastructure")
        if not runtime.certified or not all(runtime.validation.model_dump().values()):
            raise ValueError("infrastructure requires a certified runtime")
        for uri, root in ((dataset.root_uri, value.dataset_root_uri),
                          (job.staging_uri, value.artifact_root_uri)):
            if uri != root and not uri.startswith(root.rstrip("/") + "/"):
                raise ValueError("job or dataset location is outside observed infrastructure")


def serialize_capabilities(value: InfrastructureCapabilities) -> bytes:
    value = InfrastructureCapabilities.model_validate(value.model_dump(mode="json"))
    return canonical_bytes({"capabilities": value.model_dump(mode="json"), "sha256": value.sha256})


def load_capabilities_from_env() -> InfrastructureCapabilities | None:
    path = os.getenv("DEFECT_INFRA_CAPABILITIES_FILE")
    expected = os.getenv("DEFECT_INFRA_CAPABILITIES_SHA256")
    if not path and not expected:
        return None
    if not path or not expected or not re.fullmatch(SHA256, expected):
        raise ValueError("configure both infrastructure capability file and pinned SHA-256")
    if path.startswith("gs://"):
        from google.cloud import storage
        parsed = urlparse(path)
        payload = storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_as_bytes()
    else:
        payload = Path(path).read_bytes()
    envelope = json.loads(payload)
    if not isinstance(envelope, dict) or set(envelope) != {"capabilities", "sha256"}:
        raise ValueError("invalid infrastructure capability envelope")
    value = InfrastructureCapabilities.model_validate(envelope["capabilities"])
    if envelope["sha256"] != value.sha256 or value.sha256 != expected:
        raise ValueError("infrastructure capability fingerprint mismatch")
    return value
