"""Run lifecycle orchestration. External systems are passed as narrow adapters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from defect_platform.contracts import (
    CertifiedRuntime, DatasetVersion, ExperimentConfig, RunRecord, RunState, VertexJobConfig,
)
from defect_platform.control.store import RunStore

log = logging.getLogger(__name__)


class RuntimeCatalog(Protocol):
    def get_certified(self, runtime_id: str) -> CertifiedRuntime | None: ...


class DatasetCatalog(Protocol):
    def get(self, version_id: str) -> DatasetVersion | None: ...


class VertexSubmitter(Protocol):
    def submit(self, *, run_id: str, experiment: ExperimentConfig, dataset: DatasetVersion,
               runtime: CertifiedRuntime, job: VertexJobConfig, output_uri: str,
               classes: list[str], mlflow_run_id: str | None = None,
               mlflow_tracking_uri: str | None = None) -> str: ...


class WorkflowStarter(Protocol):
    def start(self, *, run_id: str, payload: dict[str, Any]) -> str: ...


class ExperimentTracker(Protocol):
    def start(self, *, run: RunRecord, experiment: ExperimentConfig, dataset: DatasetVersion,
              runtime: CertifiedRuntime, config: dict[str, Any]) -> str: ...
    def finish(self, run: RunRecord, metrics: dict[str, float] | None = None) -> None: ...


def classify_failure(error: BaseException) -> tuple[str, str]:
    """Stable, operator-facing owning-layer classification for a failed stage."""
    message = str(error).strip() or error.__class__.__name__
    text = message.lower()
    if any(x in text for x in ("quota", "accelerator", "gpu", "machine type")):
        code = "VERTEX_CAPACITY"
    elif any(x in text for x in ("permission", "forbidden", "unauthorized", "403")):
        code = "CLOUD_PERMISSION"
    elif any(x in text for x in ("not found", "404", "dataset", "gs://")):
        code = "DATA_OR_ARTIFACT"
    elif any(x in text for x in ("mlflow", "tracking uri", "experiment tracker")):
        code = "MLFLOW_TRACKING"
    elif any(x in text for x in ("workflow", "orchestrat")):
        code = "WORKFLOW"
    else:
        code = "CONTROL_PLANE"
    return code, message[:2000]


class RunController:
    def __init__(self, *, store: RunStore, runtimes: RuntimeCatalog, datasets: DatasetCatalog,
                 vertex: VertexSubmitter | None = None, workflows: WorkflowStarter | None = None,
                 tracker: ExperimentTracker | None = None, clock=lambda: datetime.now(UTC)):
        self.store, self.runtimes, self.datasets = store, runtimes, datasets
        self.vertex, self.workflows, self.tracker, self.clock = vertex, workflows, tracker, clock

    @staticmethod
    def estimate_cost(job: VertexJobConfig) -> float:
        return round(job.max_run_hours * job.estimated_hourly_usd, 2)

    def submit(self, *, experiment: ExperimentConfig, job: VertexJobConfig,
               idempotency_key: str, runtime_override: CertifiedRuntime | None = None,
               dataset_override: DatasetVersion | None = None,
               classes: list[str] | None = None) -> tuple[RunRecord, bool]:
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("a non-empty idempotency key of at most 200 characters is required")
        if job.max_run_cost_usd <= 0:
            raise ValueError("a positive per-run cost cap must be configured")
        estimated = self.estimate_cost(job)
        if estimated > job.max_run_cost_usd:
            raise ValueError(f"estimated cost ${estimated:.2f} exceeds configured run cap ${job.max_run_cost_usd:.2f}")
        runtime = runtime_override or self.runtimes.get_certified(experiment.runtime_id)
        if runtime is None or not runtime.certified or "@sha256:" not in runtime.image_digest:
            raise ValueError(f"runtime {experiment.runtime_id!r} is not certified with an immutable image digest")
        dataset = dataset_override or self.datasets.get(experiment.dataset_version_id)
        if dataset is None:
            raise ValueError(f"dataset version not found: {experiment.dataset_version_id}")
        if dataset.object_slug != experiment.object_slug:
            raise ValueError("dataset object does not match experiment object")
        if dataset.version_id != experiment.dataset_version_id:
            raise ValueError("dataset version does not match experiment dataset_version_id")
        if runtime.runtime_id != experiment.runtime_id:
            raise ValueError("runtime record does not match experiment runtime_id")
        if not classes or len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("canonical object classes must be provided as two or more unique labels")
        from defect_platform.dataset import verify_dataset_version
        verify_dataset_version(dataset, expected_classes=classes)
        now = self.clock()
        run_id = str(uuid.uuid4())
        output_uri = f"gs://{job.staging_uri.removeprefix('gs://').rstrip('/')}/runs/{run_id}"
        run = RunRecord(run_id=run_id, object_slug=experiment.object_slug,
                        experiment_id=experiment.experiment_id,
                        dataset_version_id=dataset.version_id, runtime_id=runtime.runtime_id,
                        state=RunState.PENDING, created_at=now, output_uri=output_uri)
        payload = {"idempotency_key": idempotency_key,
                   "experiment": experiment.model_dump(mode="json"),
                   "job": job.model_dump(mode="json"), "runtime": runtime.model_dump(mode="json"),
                   "dataset": dataset.model_dump(mode="json"), "estimated_cost_usd": estimated,
                   "classes": classes, "output_uri": output_uri}
        # Generated IDs and output locations must not change the fingerprint on retry.
        fingerprint_data = {key: value for key, value in payload.items()
                            if key not in {"output_uri", "idempotency_key"}}
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        record, created = self.store.create(run, fingerprint, payload)
        if not created:
            return record, False
        try:
            if self.tracker:
                mlflow_run_id = self.tracker.start(run=record, experiment=experiment, dataset=dataset,
                                                   runtime=runtime, config=payload)
                record = record.model_copy(update={"mlflow_run_id": mlflow_run_id})
                self.store.update(record)
            if self.workflows:
                # Publish run ID and submitted state before the workflow can call back.
                record = record.model_copy(update={"state": RunState.SUBMITTED})
                self.store.update(record)
                # Workflow execution is kicked off only after the immutable run record exists.
                self.workflows.start(run_id=record.run_id, payload=payload)
            elif self.vertex:
                job_name = self.vertex.submit(run_id=record.run_id, experiment=experiment,
                                              dataset=dataset, runtime=runtime, job=job,
                                              output_uri=output_uri, classes=classes,
                                              mlflow_run_id=record.mlflow_run_id,
                                              mlflow_tracking_uri=getattr(self.tracker, "tracking_uri", None))
                record = record.model_copy(update={"state": RunState.SUBMITTED,
                                                   "vertex_job_name": job_name})
            else:
                raise RuntimeError("no workflow starter or Vertex submitter is configured")
            self.store.update(record)
            return record, True
        except Exception as exc:
            code, message = classify_failure(exc)
            record = record.model_copy(update={"state": RunState.FAILED, "failure_code": code,
                                               "failure_message": message})
            self.store.update(record)
            if self.tracker:
                try:
                    self.tracker.finish(record)
                except Exception:
                    log.exception("could not mark failed MLflow run %s", record.mlflow_run_id)
            raise

    def dispatch_vertex(self, run_id: str) -> str:
        """Workflow stage to submit a pending run; repeated calls return its existing job reference."""
        current = self.get(run_id)
        if current.vertex_job_name and current.state in {RunState.SUBMITTED, RunState.RUNNING}:
            return current.vertex_job_name
        if current.state not in {RunState.PENDING, RunState.PREPARING, RunState.SUBMITTED}:
            raise ValueError(f"cannot dispatch Vertex job from {current.state}")
        if self.vertex is None:
            raise RuntimeError("Vertex submitter is not configured")
        payload = self.store.get_payload(run_id)
        if payload is None:
            raise KeyError(f"stored request payload not found for run: {run_id}")
        experiment = ExperimentConfig.model_validate(payload["experiment"])
        job = VertexJobConfig.model_validate(payload["job"])
        dataset = DatasetVersion.model_validate(payload["dataset"])
        runtime = CertifiedRuntime.model_validate(payload["runtime"])
        name = self.vertex.submit(run_id=run_id, experiment=experiment, dataset=dataset,
                                  runtime=runtime, job=job, output_uri=payload["output_uri"],
                                  classes=payload["classes"], mlflow_run_id=current.mlflow_run_id,
                                  mlflow_tracking_uri=getattr(self.tracker, "tracking_uri", None))
        self.update_state(run_id, RunState.SUBMITTED, vertex_job_name=name)
        return name

    def get(self, run_id: str) -> RunRecord:
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(f"run not found: {run_id}")
        return record

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        return self.store.list(object_slug)

    def update_state(self, run_id: str, state: RunState, *, vertex_job_name: str | None = None,
                     failure: BaseException | None = None, logs_uri: str | None = None,
                     failure_code: str | None = None, failure_message: str | None = None,
                     metrics: dict[str, float] | None = None) -> RunRecord:
        current = self.get(run_id)
        allowed = {
            RunState.PENDING: {RunState.PREPARING, RunState.SUBMITTED, RunState.FAILED, RunState.CANCELED},
            RunState.PREPARING: {RunState.SUBMITTED, RunState.FAILED, RunState.CANCELED},
            RunState.SUBMITTED: {RunState.RUNNING, RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELED},
            RunState.RUNNING: {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELED},
            RunState.SUCCEEDED: set(), RunState.FAILED: set(), RunState.CANCELED: set(),
        }
        if state != current.state and state not in allowed[current.state]:
            raise ValueError(f"invalid run transition: {current.state} -> {state}")
        fields: dict[str, Any] = {"state": state}
        if vertex_job_name is not None: fields["vertex_job_name"] = vertex_job_name
        if logs_uri is not None: fields["logs_uri"] = logs_uri
        if failure is not None:
            fields["failure_code"], fields["failure_message"] = classify_failure(failure)
        elif failure_code or failure_message:
            code, message = (failure_code or "WORKFLOW"), (failure_message or "")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,62}", code):
                raise ValueError("failure_code must be an uppercase stable identifier")
            fields["failure_code"], fields["failure_message"] = code, message[:2000]
        updated = current.model_copy(update=fields)
        self.store.update(updated)
        if state in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELED} and self.tracker:
            transitioned = current.state != state
            try:
                if transitioned:
                    self.tracker.finish(updated, metrics)
                if state == RunState.SUCCEEDED and hasattr(self.tracker, "register_candidate"):
                    self.tracker.register_candidate(updated)
                if current.failure_code == "MLFLOW_MODEL_REGISTRATION":
                    updated = updated.model_copy(update={"failure_code": None, "failure_message": None})
                    self.store.update(updated)
            except Exception as exc:
                owner_code, message = classify_failure(exc)
                code = "MLFLOW_MODEL_REGISTRATION" if state == RunState.SUCCEEDED else "MLFLOW_FINALIZATION"
                updated = updated.model_copy(update={"failure_code": code,
                                                     "failure_message": f"{owner_code}: {message}"})
                self.store.update(updated)
                log.exception("terminal MLflow processing failed for run %s", run_id)
        return updated


class VertexAiplatformSubmitter:
    """Create a server-side CustomJob and return its resource name immediately."""
    def __init__(self, project: str | None = None, location: str | None = None,
                 client_factory=None):
        self.project, self.location = project, location
        self.client_factory = client_factory

    def submit(self, *, run_id: str, experiment: ExperimentConfig, dataset: DatasetVersion,
               runtime: CertifiedRuntime, job: VertexJobConfig, output_uri: str,
               classes: list[str], mlflow_run_id: str | None = None,
               mlflow_tracking_uri: str | None = None) -> str:
        try:
            from google.cloud import aiplatform_v1
        except ImportError as exc:
            raise RuntimeError("Vertex submission requires the optional cloud dependencies") from exc
        project, location = self.project or job.project, self.location or job.region
        safe_id = re.sub(r"[^a-z0-9-]", "-", run_id.lower())[:50].strip("-")
        display_name = f"defect-{experiment.object_slug}-{safe_id}"
        client = (self.client_factory(location) if self.client_factory else
                  aiplatform_v1.JobServiceClient(
                      client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}))
        parent = f"projects/{project}/locations/{location}"
        # Recover a paid job created immediately before a process crash.
        existing = list(client.list_custom_jobs(request={
            "parent": parent, "filter": f'display_name="{display_name}"',
        }))
        names = {str(item.name) for item in existing if item.display_name == display_name}
        if len(names) == 1:
            return names.pop()
        if names:
            raise RuntimeError(f"multiple Vertex jobs already exist for run {run_id}: {sorted(names)}")
        request_uri = _upload_vertex_request(
            job.staging_uri, run_id, {
                "experiment": experiment.model_dump(mode="json"),
                "dataset": dataset.model_dump(mode="json"), "classes": classes,
                "output_uri": output_uri, "mlflow_run_id": mlflow_run_id,
                "mlflow_tracking_uri": mlflow_tracking_uri,
            }
        )
        job_spec = {
            "worker_pool_specs": [{
                "machine_spec": {"machine_type": job.machine_type,
                                 "accelerator_type": job.accelerator_type,
                                 "accelerator_count": job.accelerator_count},
                "replica_count": 1,
                "container_spec": {"image_uri": runtime.image_digest,
                    "command": ["python", "-m", "defect_platform.trainer.runner"],
                    "args": ["--request", request_uri],
                    "env": [{"name": "DEFECT_RUN_ID", "value": run_id}]},
            }],
            "service_account": job.service_account,
            "scheduling": {"timeout": f"{max(1, int(job.max_run_hours * 3600))}s"},
        }
        if job.network:
            job_spec["network"] = job.network
        custom = aiplatform_v1.CustomJob(
            display_name=display_name, job_spec=job_spec,
            labels={"defect-run-id": safe_id, "object": experiment.object_slug,
                    "runtime": re.sub(r"[^a-z0-9_-]", "-", runtime.runtime_id.lower())[:63]},
        )
        created = client.create_custom_job(parent=parent, custom_job=custom)
        if not created.name:
            raise RuntimeError("Vertex created a CustomJob without returning its resource name")
        return str(created.name)


def _upload_vertex_request(staging_uri: str, run_id: str, request: dict[str, Any]) -> str:
    """Persist the exact trainer request before CustomJob submission."""
    uri = staging_uri.rstrip("/") + f"/requests/{run_id}.json"
    if not uri.startswith("gs://"):
        raise ValueError("Vertex staging_uri must be a gs:// bucket or prefix")
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("Vertex trainer request upload requires optional cloud dependencies") from exc
    bucket, _, key = uri[5:].partition("/")
    blob = storage.Client().bucket(bucket).blob(key)
    encoded = json.dumps(request, sort_keys=True)
    try:
        blob.upload_from_string(encoded, content_type="application/json", if_generation_match=0)
    except Exception as exc:
        # A retry after a process restart is valid only if the immutable request is identical.
        try:
            if blob.download_as_text() != encoded:
                raise ValueError(f"immutable trainer request already exists with different contents: {uri}") from exc
        except ValueError:
            raise
        except Exception:
            raise exc
    return uri


class MLflowTracker:
    """Lazy MLflow adapter which records immutable dataset/runtime lineage."""
    def __init__(self, tracking_uri: str, experiment_name: str = "defect-classification",
                 *, iam_auth: bool | None = None, token_provider=None):
        if not tracking_uri:
            raise ValueError("MLflow tracking URI must be configured")
        self.tracking_uri, self.experiment_name = tracking_uri, experiment_name
        self.iam_auth = iam_auth if iam_auth is not None else os.environ.get("DEFECT_MLFLOW_IAM_AUTH", "").lower() in {"1", "true", "yes"}
        self.token_provider = token_provider

    def start(self, *, run: RunRecord, experiment: ExperimentConfig, dataset: DatasetVersion,
              runtime: CertifiedRuntime, config: dict[str, Any]) -> str:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow tracking requires optional cloud dependencies") from exc
        mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        with self._auth():
            experiment_obj = client.get_experiment_by_name(self.experiment_name)
            experiment_id = (experiment_obj.experiment_id if experiment_obj
                             else client.create_experiment(self.experiment_name))
        tags = {"platform.run_id": run.run_id, "object.slug": run.object_slug,
                "dataset.version": dataset.version_id, "dataset.sha256": dataset.sha256,
                "runtime.id": runtime.runtime_id, "runtime.image_digest": runtime.image_digest}
        with self._auth():
            active = client.create_run(experiment_id=experiment_id, tags=tags,
                                       run_name=run.experiment_id)
        params = {"epochs": experiment.epochs, "batch_size": experiment.batch_size,
                  "learning_rate": experiment.learning_rate, "output_uri": run.output_uri or ""}
        for name, value in params.items():
            with self._auth(): client.log_param(active.info.run_id, name, value)
        with tempfile.TemporaryDirectory(prefix="defect-run-config-") as tempdir:
            path = Path(tempdir) / "run-config.json"
            path.write_text(json.dumps(config, sort_keys=True, indent=2))
            with self._auth(): client.log_artifact(active.info.run_id, str(path))
        return active.info.run_id

    def finish(self, run: RunRecord, metrics: dict[str, float] | None = None) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow tracking requires optional cloud dependencies") from exc
        if run.mlflow_run_id:
            client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
            if metrics:
                for name, value in metrics.items():
                    with self._auth(): client.log_metric(run.mlflow_run_id, name, value)
            with self._auth(): client.set_tag(run.mlflow_run_id, "platform.state", run.state.value)
            if run.failure_code:
                with self._auth(): client.set_tag(run.mlflow_run_id, "platform.failure_code", run.failure_code)
            status = "FINISHED" if run.state == RunState.SUCCEEDED else "KILLED" if run.state == RunState.CANCELED else "FAILED"
            with self._auth(): client.set_terminated(run.mlflow_run_id, status=status)

    def register_candidate(self, run: RunRecord) -> str:
        """Register `model/` from the trainer MLflow run, preserving checkpoint and class map."""
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow model registry requires optional cloud dependencies") from exc
        if not run.mlflow_run_id:
            raise ValueError("successful run has no MLflow run id")
        model_name = f"defect-{run.object_slug}"
        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        with self._auth():
            versions = client.search_model_versions(f"name='{model_name}'")
        version = next((item for item in versions if item.run_id == run.mlflow_run_id), None)
        if version is None:
            with self._auth():
                try:
                    client.create_registered_model(model_name)
                except Exception as exc:
                    if getattr(exc, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
                        raise
                source_run = client.get_run(run.mlflow_run_id)
                version = client.create_model_version(
                    name=model_name,
                    source=source_run.info.artifact_uri.rstrip("/") + "/model",
                    run_id=run.mlflow_run_id,
                    tags={"platform.run_id": run.run_id,
                          "dataset.version": run.dataset_version_id,
                          "runtime.id": run.runtime_id},
                )
        with self._auth():
            client.set_model_version_tag(model_name, version.version, "platform.run_id", run.run_id)
            client.set_model_version_tag(model_name, version.version, "dataset.version", run.dataset_version_id)
            client.set_model_version_tag(model_name, version.version, "runtime.id", run.runtime_id)
        return f"models:/{model_name}/{version.version}"

    @contextmanager
    def _auth(self):
        if self.token_provider:
            previous_token = os.environ.get("MLFLOW_TRACKING_TOKEN")
            previous_iam = os.environ.get("DEFECT_MLFLOW_IAM_AUTH")
            os.environ["MLFLOW_TRACKING_TOKEN"] = self.token_provider(self.tracking_uri)
            os.environ["DEFECT_MLFLOW_IAM_AUTH"] = "1"
            try:
                yield
            finally:
                if previous_token is None: os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
                else: os.environ["MLFLOW_TRACKING_TOKEN"] = previous_token
                if previous_iam is None: os.environ.pop("DEFECT_MLFLOW_IAM_AUTH", None)
                else: os.environ["DEFECT_MLFLOW_IAM_AUTH"] = previous_iam
            return
        previous = os.environ.get("DEFECT_MLFLOW_IAM_AUTH")
        if self.iam_auth:
            os.environ["DEFECT_MLFLOW_IAM_AUTH"] = "1"
        try:
            from defect_platform.mlflow_auth import mlflow_tracking_auth
            with mlflow_tracking_auth(self.tracking_uri):
                yield
        finally:
            if previous is None:
                os.environ.pop("DEFECT_MLFLOW_IAM_AUTH", None)
            else:
                os.environ["DEFECT_MLFLOW_IAM_AUTH"] = previous


class GoogleWorkflowsStarter:
    """Starts a configured Cloud Workflow and returns immediately with its execution name."""
    def __init__(self, *, project: str, region: str, workflow_name: str,
                 control_service_url: str | None = None, max_polls: int = 240):
        if not all((project, region, workflow_name)):
            raise ValueError("GCP project, region, and workflow name are required")
        self.project, self.region, self.workflow_name = project, region, workflow_name
        self.control_service_url, self.max_polls = (control_service_url.rstrip("/") if control_service_url else None), max_polls

    def start(self, *, run_id: str, payload: dict[str, Any]) -> str:
        try:
            from google.cloud import workflows_v1
        except ImportError as exc:
            raise RuntimeError("Cloud Workflows requires optional cloud dependencies") from exc
        client = workflows_v1.ExecutionsClient()
        parent = f"projects/{self.project}/locations/{self.region}/workflows/{self.workflow_name}"
        # The complete request is already in Cloud SQL; only the run locator
        # and bounded polling parameters belong in the workflow execution.
        args = {"run_id": run_id, "region": self.region, "max_polls": self.max_polls}
        execution = workflows_v1.Execution(argument=json.dumps(args))
        result = client.create_execution(request={"parent": parent, "execution": execution})
        return result.name
