"""FastAPI surface for Cloud Run and Cloud Workflows callbacks.

Cloud Run IAM should protect this service. The controller remains dependency-injected.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from defect_platform.contracts import CertifiedRuntime, DatasetVersion, ExperimentConfig, RunState, VertexJobConfig
from defect_platform.control.controller import (GoogleWorkflowsStarter, MLflowTracker, RunController,
                                                VertexAiplatformSubmitter)
from defect_platform.control.store import run_store_from_env
from defect_platform.telemetry import configure_logging


class SubmitBody(BaseModel):
    experiment: ExperimentConfig
    job: VertexJobConfig
    dataset: DatasetVersion
    runtime: CertifiedRuntime
    classes: list[str]
    idempotency_key: str


class EventBody(BaseModel):
    state: RunState
    vertex_job_name: str | None = None
    logs_uri: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    metrics: dict[str, float] | None = None

def create_app(controller: RunController):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("API service requires the optional FastAPI dependency") from exc

    app = FastAPI(title="Defect training control plane", version="1")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=202)
    def submit(body: SubmitBody) -> dict[str, Any]:
        try:
            run, created = controller.submit(experiment=body.experiment, job=body.job,
                                             idempotency_key=body.idempotency_key,
                                             runtime_override=body.runtime,
                                             dataset_override=body.dataset,
                                             classes=body.classes)
            return {"run": run.model_dump(mode="json"), "created": created}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/runs")
    def list_runs(object_slug: str | None = None) -> list[dict[str, Any]]:
        return [run.model_dump(mode="json") for run in controller.list(object_slug)]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return controller.get(run_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/vertex-submit")
    def vertex_submit(run_id: str) -> dict[str, str]:
        try:
            return {"vertex_job_name": controller.dispatch_vertex(run_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/events")
    def event(run_id: str, body: EventBody) -> dict[str, Any]:
        try:
            run = controller.update_state(run_id, body.state,
                                          vertex_job_name=body.vertex_job_name,
                                          failure_code=body.failure_code,
                                          failure_message=body.failure_message,
                                          logs_uri=body.logs_uri,
                                          metrics=body.metrics)
            return run.model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


class _InlineRuntimeCatalog:
    def get_certified(self, _runtime_id):
        return None


class _InlineDatasetCatalog:
    def get(self, _version_id):
        return None


def create_default_app():
    """Build the Cloud Run app from environment config without importing cloud SDKs."""
    import os
    tracking_uri = os.environ.get("DEFECT_MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_URI")
    tracker = MLflowTracker(tracking_uri, iam_auth=True) if tracking_uri else None
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "")
    workflow_name = os.environ.get("DEFECT_WORKFLOW_NAME", "")
    control_url = os.environ.get("CONTROL_URL", "")
    workflows = (GoogleWorkflowsStarter(project=project, region=region,
                 workflow_name=workflow_name, control_service_url=control_url or None)
                 if workflow_name else None)
    configure_logging()
    controller = RunController(store=run_store_from_env(), runtimes=_InlineRuntimeCatalog(),
                               datasets=_InlineDatasetCatalog(),
                               vertex=VertexAiplatformSubmitter(), workflows=workflows,
                               tracker=tracker)
    return create_app(controller)


class _LazyASGIApp:
    """Keep package import lightweight; initialize FastAPI only when serving starts."""
    def __init__(self):
        self._app = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            self._app = create_default_app()
        await self._app(scope, receive, send)


app = _LazyASGIApp()
