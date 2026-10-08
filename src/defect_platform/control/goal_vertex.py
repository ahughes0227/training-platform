"""Run each goal experiment as a Vertex AI CustomJob instead of on this machine.

``VertexExecutor`` implements the goal runner's ``RunExecutor``: it uploads the
trainer request, creates one CustomJob from an immutable image digest, waits
for it to finish, and copies the run's ``evaluation.json`` and model back into
the local run directory, so goal scoring and reports work unchanged.

A CPU machine is allowed (no accelerator), so the small demo goal runs without
a GPU quota. Requests include the certified runtime when one is supplied. An
uncertified development image is accepted only when the config says so, and
the goal report records it, because such a model is not releasable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..contracts import CertifiedRuntime, DatasetVersion, ExperimentConfig, StrictModel
from .controller import _upload_vertex_request

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                   "JOB_STATE_EXPIRED"}


class VertexGoalConfig(StrictModel):
    project: str = Field(min_length=1)
    region: str = Field(min_length=1)
    service_account: str = Field(min_length=1)
    staging_uri: str = Field(pattern=r"^gs://[^/]+")
    image_digest: str = Field(pattern=r"^[^\s@]+@sha256:[a-f0-9]{64}$")
    machine_type: str = "n1-standard-4"
    accelerator_type: str | None = None
    accelerator_count: int = Field(default=0, ge=0)
    max_run_hours: float = Field(default=2, gt=0)
    poll_seconds: float = Field(default=30, gt=0)
    runtime: CertifiedRuntime | None = None
    # A development image skips certification. Its models are for evaluation only.
    allow_uncertified_image: bool = False

    @model_validator(mode="after")
    def accelerator_pair(self) -> VertexGoalConfig:
        if (self.accelerator_type is None) != (self.accelerator_count == 0):
            raise ValueError("accelerator_type and accelerator_count must be set together")
        if self.runtime is not None:
            if not self.runtime.certified:
                raise ValueError("runtime must be a certified record")
            if self.runtime.image_digest != self.image_digest:
                raise ValueError("image_digest must match the certified runtime's digest")
        elif not self.allow_uncertified_image:
            raise ValueError("supply a certified runtime record, or set "
                             "allow_uncertified_image for a development run")
        return self

    @property
    def certified(self) -> bool:
        return self.runtime is not None


def _job_client(region: str):
    from google.cloud import aiplatform_v1

    return aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"})


def download_prefix(uri: str, destination: Path) -> None:
    """Copy every object under a gs:// prefix into a local directory."""
    from google.cloud import storage

    bucket, _, prefix = uri[5:].partition("/")
    prefix = prefix.rstrip("/") + "/"
    found = False
    for blob in storage.Client().list_blobs(bucket, prefix=prefix):
        found = True
        target = destination / blob.name[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
    if not found:
        raise FileNotFoundError(f"the run wrote nothing to {uri}")


def upload_directory(source: Path, uri: str) -> None:
    from google.cloud import storage

    bucket_name, _, prefix = uri[5:].partition("/")
    bucket = storage.Client().bucket(bucket_name)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            key = f"{prefix.rstrip('/')}/{path.relative_to(source).as_posix()}".lstrip("/")
            bucket.blob(key).upload_from_filename(str(path))


class VertexExecutor:
    def __init__(self, config: VertexGoalConfig, dataset: DatasetVersion, classes: list[str], *,
                 client: Any = None, sleep: Callable[[float], None] = time.sleep,
                 upload_request: Callable[[str, str, dict], str] = _upload_vertex_request,
                 download: Callable[[str, Path], None] = download_prefix,
                 semantic_refs: dict[str, str] | None = None):
        if not dataset.root_uri.startswith("gs://"):
            raise ValueError("Vertex runs need the dataset in GCS; build it with a gs:// output_uri")
        self.config = config
        self.dataset = dataset
        self.classes = list(classes)
        self.client = client
        self.sleep = sleep
        self.upload_request = upload_request
        self.download = download
        self.semantic_refs = semantic_refs
        runtime = (f"certified runtime {config.runtime.runtime_id}" if config.runtime
                   else "an uncertified development image")
        self.description = f"Vertex AI ({config.machine_type}, {runtime})"
        self.releasable = config.certified

    def _client(self):
        if self.client is None:
            self.client = _job_client(self.config.region)
        return self.client

    def _refs(self) -> dict[str, str]:
        if self.semantic_refs is None:
            from ..dataset import load_dataset_semantics

            manifest = load_dataset_semantics(self.dataset)
            self.semantic_refs = {"dataset_semantic_sha256": manifest.sha256,
                                  "catalog_sha256": manifest.catalog.sha256}
        return self.semantic_refs

    def output_uri(self, experiment: ExperimentConfig) -> str:
        return f"{self.config.staging_uri.rstrip('/')}/runs/{experiment.experiment_id}"

    def _request(self, experiment: ExperimentConfig) -> dict[str, Any]:
        request = {"experiment": experiment.model_dump(mode="json"),
                   "dataset": self.dataset.model_dump(mode="json"),
                   "classes": self.classes, "output_uri": self.output_uri(experiment)}
        if self.config.runtime is not None:
            request["runtime"] = self.config.runtime.model_dump(mode="json")
            request["semantic_refs"] = self._refs()
        return request

    def _job_spec(self, request_uri: str, run_id: str) -> dict[str, Any]:
        config = self.config
        machine: dict[str, Any] = {"machine_type": config.machine_type}
        if config.accelerator_type:
            machine.update(accelerator_type=config.accelerator_type,
                           accelerator_count=config.accelerator_count)
        return {
            "worker_pool_specs": [{
                "machine_spec": machine, "replica_count": 1,
                "container_spec": {"image_uri": config.image_digest,
                                   "command": ["python", "-m", "defect_platform.trainer.runner"],
                                   "args": ["--request", request_uri],
                                   "env": [{"name": "DEFECT_RUN_ID", "value": run_id}]},
            }],
            "service_account": config.service_account,
            "scheduling": {"timeout": f"{max(1, int(config.max_run_hours * 3600))}s"},
        }

    def _start(self, experiment: ExperimentConfig) -> str:
        run_id = experiment.experiment_id
        if not experiment.model.weights_uri.startswith("gs://"):
            raise ValueError("Vertex runs need the backbone weights in GCS (a gs:// weights_uri)")
        display_name = "defect-goal-" + re.sub(r"[^a-z0-9-]", "-", run_id.lower())[:100]
        client = self._client()
        parent = f"projects/{self.config.project}/locations/{self.config.region}"
        # A goal resumed after a crash re-attaches to its live or finished job
        # instead of paying for a second one.
        for job in client.list_custom_jobs(request={
                "parent": parent, "filter": f'display_name="{display_name}"'}):
            if job.display_name == display_name and _state(job) not in (
                    TERMINAL_STATES - {"JOB_STATE_SUCCEEDED"}):
                return str(job.name)
        request_uri = self.upload_request(self.config.staging_uri, run_id, self._request(experiment))
        from google.cloud import aiplatform_v1

        created = client.create_custom_job(parent=parent, custom_job=aiplatform_v1.CustomJob(
            display_name=display_name, job_spec=self._job_spec(request_uri, run_id),
            labels={"defect-goal-run": re.sub(r"[^a-z0-9_-]", "-", run_id.lower())[:63]}))
        if not created.name:
            raise RuntimeError("Vertex created a CustomJob without returning its resource name")
        return str(created.name)

    def run(self, experiment: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
        job_name = self._start(experiment)
        logger.info("vertex_goal_run_submitted", extra={"stage": "vertex_goal_run_submitted",
                                                        "run_id": experiment.experiment_id,
                                                        "vertex_job_name": job_name})
        deadline = time.monotonic() + self.config.max_run_hours * 3600 + 15 * 60
        while True:
            job = self._client().get_custom_job(name=job_name)
            state = _state(job)
            if state in TERMINAL_STATES:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"{job_name} did not finish within its run limit")
            self.sleep(self.config.poll_seconds)
        if state != "JOB_STATE_SUCCEEDED":
            message = getattr(getattr(job, "error", None), "message", "") or "no error message"
            raise RuntimeError(f"Vertex job {job_name} ended {state}: {message}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.download(self.output_uri(experiment), run_dir)
        report = json.loads((run_dir / "evaluation.json").read_text())
        report["vertex_job_name"] = job_name
        return report


def _state(job: Any) -> str:
    state = getattr(job, "state", "")
    return getattr(state, "name", None) or str(state)
