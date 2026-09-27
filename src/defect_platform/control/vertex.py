"""Vertex CustomJob adapters for runtime certification handshakes."""

from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Any


class VertexHandshakeError(RuntimeError):
    """A certification job did not complete or did not produce a valid result."""


class VertexAdapter:
    """Submit a short GPU/GCS probe job, wait for completion, and read its GCS result.

    SDK clients may be injected to keep unit tests offline. The trainer probe writes JSON at
    `environment.DEFECT_PLATFORM_GCS_RESULT_URI` with at least `gpu_count`, `gcs_read`,
    `gcs_write`, and `image_digest`.
    """

    def __init__(self, *, custom_job_factory=None, storage_client=None):
        self.custom_job_factory = custom_job_factory
        self.storage_client = storage_client

    def submit_handshake_and_wait(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = ("project", "region", "machine_type", "accelerator_type", "accelerator_count",
                    "service_account", "staging_uri", "image_uri", "gcs_probe_uri",
                    "timeout_seconds", "environment", "command")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Vertex handshake payload is missing: {', '.join(missing)}")
        result_uri = payload["environment"].get("DEFECT_PLATFORM_GCS_RESULT_URI")
        expected_digest = payload["environment"].get("DEFECT_PLATFORM_IMAGE_DIGEST")
        digest_suffix = payload["image_uri"].split("@", 1)[1]
        if not result_uri or not result_uri.startswith("gs://"):
            raise ValueError("environment.DEFECT_PLATFORM_GCS_RESULT_URI must be a gs:// URI")
        if "@sha256:" not in payload["image_uri"]:
            raise ValueError("Vertex handshake must use an immutable container image digest")
        if expected_digest and expected_digest != digest_suffix:
            raise ValueError("handshake image digest and image URI do not match")
        job = self._create_job(payload)
        try:
            job.run(service_account=payload["service_account"], network=payload.get("network"),
                    sync=True, timeout=payload["timeout_seconds"],
                    restart_job_on_worker_restart=False)
        except Exception as exc:
            raise VertexHandshakeError(f"Vertex handshake job failed while waiting: {exc}") from exc
        state = str(getattr(job, "state", "")).upper()
        if not any(value in state for value in ("SUCCEEDED", "JOB_STATE_SUCCEEDED")):
            error = getattr(job, "error", None)
            raise VertexHandshakeError(f"Vertex handshake did not succeed (state={state or 'unknown'}): {error or ''}")
        result = self._read_gcs_json(result_uri, project=payload["project"])
        missing_result = [name for name in ("gpu_count", "gcs_read", "gcs_write", "image_digest") if name not in result]
        if missing_result:
            raise VertexHandshakeError(f"handshake result is missing fields: {', '.join(missing_result)}")
        if not isinstance(result["gpu_count"], int) or result["gpu_count"] < int(payload["accelerator_count"]):
            raise VertexHandshakeError("handshake result reports fewer GPUs than requested")
        if result["gcs_read"] is not True or result["gcs_write"] is not True:
            raise VertexHandshakeError("handshake did not verify both GCS read and write")
        if result["image_digest"] != digest_suffix:
            raise VertexHandshakeError("handshake result image digest does not match requested image")
        if not result.get("gpu_tensor_operation"):
            raise VertexHandshakeError("handshake did not complete a GPU tensor operation")
        return result

    def _create_job(self, payload):
        factory = self.custom_job_factory
        if factory is None:
            try:
                from google.cloud import aiplatform
            except ImportError as exc:
                raise RuntimeError("Vertex handshake requires optional cloud dependencies") from exc
            factory = aiplatform.CustomJob
        environment = [{"name": key, "value": str(value)} for key, value in payload["environment"].items()]
        labels = {"purpose": "runtime-handshake"}
        spec = [{
            "machine_spec": {"machine_type": payload["machine_type"],
                             "accelerator_type": payload["accelerator_type"],
                             "accelerator_count": int(payload["accelerator_count"])},
            "replica_count": 1,
            "container_spec": {"image_uri": payload["image_uri"],
                               "command": payload["command"], "env": environment},
        }]
        kwargs = {"display_name": "defect-runtime-handshake",
                  "project": payload["project"], "location": payload["region"],
                  "worker_pool_specs": spec, "staging_bucket": payload["staging_uri"],
                  "labels": labels}
        return factory(**kwargs)

    def _read_gcs_json(self, uri: str, *, project: str) -> dict[str, Any]:
        parsed = urlparse(uri)
        if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"invalid GCS result URI: {uri}")
        client = self.storage_client
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError("Reading handshake results requires optional GCS dependencies") from exc
            client = storage.Client(project=project)
        content = client.bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_as_text()
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise VertexHandshakeError(f"handshake result at {uri} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise VertexHandshakeError(f"handshake result at {uri} must be a JSON object")
        return value
