"""FastAPI surface for Cloud Run and Cloud Workflows callbacks.

Cloud Run IAM should protect this service. The controller remains dependency-injected.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from defect_platform.catalog_store import (
    ReferenceCatalogStore,
    catalog_store_from_env,
    reference_catalog_store_from_env,
)
from defect_platform.contracts import (
    CertifiedRuntime,
    ComponentDefinition,
    ComponentRef,
    DatasetVersion,
    ExperimentConfig,
    ModelSpec,
    OperatorProfile,
    OperatorProfileRef,
    Preset,
    PresetRef,
    Reference,
    RunState,
    SchemaRegistry,
    VertexJobConfig,
)
from defect_platform.control.batch import expand_matrix
from defect_platform.control.controller import (
    GoogleWorkflowsStarter,
    MLflowTracker,
    RunController,
    VertexAiplatformSubmitter,
)
from defect_platform.control.routing import ModelRoutingPolicy
from defect_platform.control.setup import (
    INTAKE_PATCH_SCHEMA_REF,
    SETUP_SCHEMA_REF,
    IntakeCoordinator,
    IntakePatch,
    SetupDraft,
)
from defect_platform.control.store import (
    IntakeSessionConflictError,
    intake_store_from_env,
    run_store_from_env,
)
from defect_platform.infrastructure_contract import load_capabilities_from_env
from defect_platform.semantics import fingerprint
from defect_platform.telemetry import configure_logging


class SubmitBody(BaseModel):
    experiment: ExperimentConfig
    job: VertexJobConfig
    dataset: DatasetVersion
    runtime: CertifiedRuntime
    classes: list[str]
    idempotency_key: str


class ReferenceSubmitBody(BaseModel):
    """Agent-facing submission: the service resolves these records authoritatively."""

    experiment: ExperimentConfig
    job: VertexJobConfig
    dataset_ref: Reference
    runtime_ref: Reference
    component_refs: list[ComponentRef] = Field(default_factory=list)
    preset_ref: PresetRef | None = None
    operator_profile_ref: OperatorProfileRef | None = None
    classes: list[str] | None = None
    idempotency_key: str


class EventBody(BaseModel):
    state: RunState | None = None
    vertex_job_name: str | None = None
    logs_uri: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    metrics: dict[str, float] | None = None
    event_id: str | None = None
    event_type: str | None = None
    payload: dict[str, Any] | None = None


class BatchBody(SubmitBody):
    matrix: dict[str, list[Any]] = Field(default_factory=dict)


class BatchPlanBody(BaseModel):
    experiment: ExperimentConfig
    job: VertexJobConfig
    idempotency_key: str = Field(min_length=1, max_length=200)
    matrix: dict[str, list[Any]] = Field(default_factory=dict)


class IntakeStartBody(BaseModel):
    request: str = Field(min_length=1, max_length=12000)
    model: str | None = None
    ttl_seconds: int = Field(default=3600, ge=1, le=604800)


class IntakeTurnBody(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    expected_revision: int = Field(ge=0)
    operation: str = "patch"


class IntakePatchBody(BaseModel):
    patch: list[dict[str, Any]]
    expected_revision: int = Field(ge=0)


def _error(
    exc: BaseException,
    *,
    code: str,
    status: int,
    retryable: bool = False,
    remediation: str | None = None,
) -> JSONResponse:
    """Create a stable machine-readable error envelope."""
    message = str(exc)
    # Admission adapters often raise a typed token in a boundary error. Keep
    # the response machine-readable without exposing a traceback or requiring
    # callers to parse prose.
    for candidate in (
        "DATASET_NOT_VERIFIED",
        "RUNTIME_DIGEST_UNTRUSTED",
        "RUNTIME_NOT_VERIFIED",
        "COMPONENT_NOT_VERIFIED",
        "PRESET_NOT_VERIFIED",
        "OPERATOR_PROFILE_NOT_VERIFIED",
        "OPERATOR_PROFILE_LIMIT",
        "OPERATOR_PROFILE_EXPIRED",
    ):
        if candidate in message:
            code = candidate
            break
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message[:2000],
                "retryable": retryable,
                "remediation": remediation
                or (
                    "retry the request"
                    if retryable
                    else "review the request and referenced records"
                ),
            }
        },
    )


def _verify_record_reference(
    record: Any, reference: Reference, *, identity: str, version: str | None = None
) -> Any:
    if record is None:
        raise ValueError(f"{identity}_NOT_VERIFIED: referenced record was not found")
    record_version = version or getattr(record, "version", None)
    record_version = record_version or getattr(record, "version_id", None)
    record_version = record_version or getattr(record, "runtime_version", None)
    if (
        reference.version is not None
        and record_version is not None
        and reference.version != record_version
    ):
        raise ValueError(f"{identity}_NOT_VERIFIED: reference version does not match record")
    valid_digests = {fingerprint(record.model_dump(mode="json"))}
    domain_digest = getattr(record, "sha256", None)
    if domain_digest:
        valid_digests.add(domain_digest.lower())
    if reference.sha256.lower() not in valid_digests:
        raise ValueError(f"{identity}_NOT_VERIFIED: reference checksum does not match record")
    return record


def _resolve_reference(
    store: ReferenceCatalogStore | None,
    reference: Reference,
    *,
    kinds: tuple[str, ...],
    model: type,
    fallback,
    identity: str,
    version: str | None = None,
) -> Any:
    record = None
    if store is not None:
        for kind in kinds:
            record = store.get_reference(
                kind, reference.id, reference.version, reference.sha256, model
            )
            if record is not None:
                break
    else:
        record = fallback(reference.id)
    return _verify_record_reference(record, reference, identity=identity, version=version)


def _apply_preset_defaults(
    experiment: ExperimentConfig, job: VertexJobConfig, preset: Preset | None
) -> tuple[ExperimentConfig, VertexJobConfig]:
    from defect_platform.control.defaults import resolve_defaults

    params = resolve_defaults()
    if preset is not None:
        params.update(preset.parameters)
    experiment_values = experiment.model_dump()
    model_values = experiment_values["model"]
    job_values = job.model_dump()
    for field in ExperimentConfig.model_fields:
        if field in params and field not in experiment.model_fields_set:
            experiment_values[field] = params[field]
    for field in ModelSpec.model_fields:
        if field in params and field not in experiment.model.model_fields_set:
            model_values[field] = params[field]
    for field in VertexJobConfig.model_fields:
        if field in params and field not in job.model_fields_set:
            job_values[field] = params[field]
    return ExperimentConfig.model_validate(experiment_values), VertexJobConfig.model_validate(
        job_values
    )


def create_app(
    controller: RunController,
    *,
    intake_store=None,
    intake_coordinator=None,
    intake_model: str | None = None,
    intake_completion=None,
    schema_registry: SchemaRegistry | None = None,
    reference_store: ReferenceCatalogStore | None = None,
):
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("API service requires the optional FastAPI dependency") from exc

    app = FastAPI(title="Defect training control plane", version="1")
    registry = schema_registry or SchemaRegistry()
    try:
        registry.get(SETUP_SCHEMA_REF, "1")
    except KeyError:
        registry.register(SETUP_SCHEMA_REF, "1", SetupDraft)
    try:
        registry.get(INTAKE_PATCH_SCHEMA_REF, "1")
    except KeyError:
        registry.register(INTAKE_PATCH_SCHEMA_REF, "1", IntakePatch)
    coordinator = intake_coordinator or (
        IntakeCoordinator(intake_store, model=intake_model, completion=intake_completion)
        if intake_store is not None
        else None
    )

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/schemas/{schema_id}", response_model=None)
    def get_schema(schema_id: str, version: str = "1") -> dict[str, Any] | JSONResponse:
        """Return a registered schema once; model turns carry only its ID/checksum."""
        try:
            return registry.get(schema_id, version).model_dump(mode="json")
        except KeyError as exc:
            return _error(exc, code="SCHEMA_NOT_FOUND", status=404)

    @app.post("/intake/sessions", response_model=None)
    def intake_start(body: IntakeStartBody) -> dict[str, Any] | JSONResponse:
        if coordinator is None:
            return _error(
                RuntimeError("intake sessions are not configured"),
                code="INTAKE_UNAVAILABLE",
                status=503,
            )
        try:
            return coordinator.start(
                body.request, model=body.model, ttl_seconds=body.ttl_seconds
            ).model_dump(mode="json")
        except (ValueError, RuntimeError) as exc:
            return _error(exc, code="INVALID_INTAKE", status=422)

    @app.get("/intake/sessions/{session_id}", response_model=None)
    def intake_get(session_id: str) -> dict[str, Any] | JSONResponse:
        if coordinator is None:
            return _error(
                RuntimeError("intake sessions are not configured"),
                code="INTAKE_UNAVAILABLE",
                status=503,
            )
        try:
            return coordinator.get(session_id).model_dump(mode="json")
        except KeyError as exc:
            return _error(exc, code="INTAKE_NOT_FOUND", status=404)

    @app.post("/intake/sessions/{session_id}/turn", response_model=None)
    def intake_turn(session_id: str, body: IntakeTurnBody) -> dict[str, Any] | JSONResponse:
        if coordinator is None:
            return _error(
                RuntimeError("intake sessions are not configured"),
                code="INTAKE_UNAVAILABLE",
                status=503,
            )
        try:
            return coordinator.turn(
                session_id,
                body.message,
                expected_revision=body.expected_revision,
                operation=body.operation,
            ).model_dump(mode="json")
        except IntakeSessionConflictError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "INTAKE_REVISION_CONFLICT",
                        "message": str(exc),
                        "retryable": False,
                        "current": exc.current.model_dump(mode="json") if exc.current else None,
                    }
                },
            )
        except KeyError as exc:
            return _error(exc, code="INTAKE_NOT_FOUND", status=404)
        except (ValueError, RuntimeError) as exc:
            return _error(exc, code="INVALID_INTAKE", status=422)

    @app.post("/intake/sessions/{session_id}/patch", response_model=None)
    def intake_patch(session_id: str, body: IntakePatchBody) -> dict[str, Any] | JSONResponse:
        if coordinator is None:
            return _error(
                RuntimeError("intake sessions are not configured"),
                code="INTAKE_UNAVAILABLE",
                status=503,
            )
        try:
            return coordinator.patch(
                session_id, body.patch, expected_revision=body.expected_revision
            ).model_dump(mode="json")
        except IntakeSessionConflictError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "INTAKE_REVISION_CONFLICT",
                        "message": str(exc),
                        "retryable": False,
                        "current": exc.current.model_dump(mode="json") if exc.current else None,
                    }
                },
            )
        except KeyError as exc:
            return _error(exc, code="INTAKE_NOT_FOUND", status=404)
        except ValueError as exc:
            return _error(exc, code="INVALID_INTAKE_PATCH", status=422)

    @app.post("/runs", status_code=202, response_model=None)
    def submit(body: SubmitBody) -> dict[str, Any] | JSONResponse:
        try:
            run, created = controller.submit(
                experiment=body.experiment,
                job=body.job,
                idempotency_key=body.idempotency_key,
                runtime_override=body.runtime,
                dataset_override=body.dataset,
                classes=body.classes,
            )
            return {"run": run.model_dump(mode="json"), "created": created}
        except (ValueError, RuntimeError) as exc:
            return _error(
                exc,
                code="INVALID_REQUEST" if isinstance(exc, ValueError) else "CONTROL_PLANE_ERROR",
                status=422 if isinstance(exc, ValueError) else 503,
            )

    @app.post("/runs/reference", status_code=202, response_model=None)
    def submit_reference(body: ReferenceSubmitBody) -> dict[str, Any] | JSONResponse:
        """Resolve dataset/runtime references before invoking normal admission."""
        try:
            if body.dataset_ref.kind not in {"dataset", "dataset_version"}:
                raise ValueError("dataset_ref must identify a dataset version")
            if body.runtime_ref.kind not in {"runtime", "certified_runtime"}:
                raise ValueError("runtime_ref must identify a certified runtime")
            dataset = _resolve_reference(
                reference_store,
                body.dataset_ref,
                kinds=("dataset", "dataset_version"),
                model=DatasetVersion,
                fallback=controller.datasets.get,
                identity="DATASET",
            )
            runtime = _resolve_reference(
                reference_store,
                body.runtime_ref,
                kinds=("runtime", "certified_runtime"),
                model=CertifiedRuntime,
                fallback=controller.runtimes.get_certified,
                identity="RUNTIME",
            )
            if "@sha256:" not in runtime.image_digest:
                raise ValueError("RUNTIME_DIGEST_UNTRUSTED: runtime is not certified")
            if runtime.runtime_id != body.experiment.runtime_id:
                raise ValueError("RUNTIME_NOT_VERIFIED: runtime does not match experiment")
            if dataset.version_id != body.experiment.dataset_version_id:
                raise ValueError("DATASET_NOT_VERIFIED: dataset does not match experiment")
            if body.dataset_ref.version not in (None, dataset.version_id):
                raise ValueError("DATASET_NOT_VERIFIED: reference version does not match dataset")
            if body.runtime_ref.version not in (None, runtime.runtime_version):
                raise ValueError("RUNTIME_NOT_VERIFIED: reference version does not match runtime")
            for component_ref in body.component_refs:
                component = _resolve_reference(
                    reference_store,
                    component_ref,
                    kinds=("component",),
                    model=ComponentDefinition,
                    fallback=lambda _id: None,
                    identity="COMPONENT",
                )
                if component.component_id != component_ref.id:
                    raise ValueError(
                        "COMPONENT_NOT_VERIFIED: component ID does not match reference"
                    )
            preset = None
            if body.preset_ref:
                from defect_platform.control.presets import builtin_presets

                builtin = next(
                    (item for item in builtin_presets() if item.preset_id == body.preset_ref.id),
                    None,
                )
                if reference_store is not None:
                    try:
                        preset = _resolve_reference(
                            reference_store,
                            body.preset_ref,
                            kinds=("preset",),
                            model=Preset,
                            fallback=lambda _id: None,
                            identity="PRESET",
                        )
                    except ValueError as exc:
                        if builtin is None or "referenced record was not found" not in str(exc):
                            raise
                        preset = _verify_record_reference(
                            builtin, body.preset_ref, identity="PRESET"
                        )
                else:
                    preset = builtin
                    if preset is None:
                        raise ValueError("PRESET_NOT_VERIFIED: preset was not found")
                    preset = _verify_record_reference(preset, body.preset_ref, identity="PRESET")
                if preset.preset_id != body.preset_ref.id:
                    raise ValueError("PRESET_NOT_VERIFIED: preset ID does not match reference")
            if body.operator_profile_ref:
                profile = _resolve_reference(
                    reference_store,
                    body.operator_profile_ref,
                    kinds=("operator_profile",),
                    model=OperatorProfile,
                    fallback=lambda _id: None,
                    identity="OPERATOR_PROFILE",
                )
                if profile.profile_id != body.operator_profile_ref.id:
                    raise ValueError(
                        "OPERATOR_PROFILE_NOT_VERIFIED: profile ID does not match reference"
                    )
                if profile.expires_at is not None and profile.expires_at <= datetime.now(UTC):
                    raise ValueError(
                        "OPERATOR_PROFILE_EXPIRED: referenced operator profile has expired"
                    )
                for field, limit in profile.limits.items():
                    requested = getattr(body.job, field, None)
                    if (
                        isinstance(limit, (int, float))
                        and isinstance(requested, (int, float))
                        and requested > limit
                    ):
                        raise ValueError(
                            f"OPERATOR_PROFILE_LIMIT: {field} exceeds the operator profile limit"
                        )
            effective, effective_job = _apply_preset_defaults(body.experiment, body.job, preset)
            effective = effective.model_copy(
                update={
                    "component_refs": tuple(body.component_refs),
                    "preset_ref": body.preset_ref,
                    "operator_profile_ref": body.operator_profile_ref,
                }
            )
            run, created = controller.submit(
                experiment=effective,
                job=effective_job,
                idempotency_key=body.idempotency_key,
                runtime_override=runtime,
                dataset_override=dataset,
                classes=body.classes,
            )
            return {"run": run.model_dump(mode="json"), "created": created}
        except (ValueError, RuntimeError) as exc:
            return _error(exc, code="REFERENCE_RESOLUTION_FAILED", status=422)

    @app.get("/runs")
    def list_runs(object_slug: str | None = None, detail: bool = False) -> list[dict[str, Any]]:
        return [
            controller.status(run.run_id, detail=detail) for run in controller.list(object_slug)
        ]

    @app.get("/runs/{run_id}", response_model=None)
    def get_run(run_id: str, detail: bool = False) -> dict[str, Any] | JSONResponse:
        try:
            return controller.status(run_id, detail=detail)
        except KeyError as exc:
            return _error(exc, code="RUN_NOT_FOUND", status=404)

    @app.get("/runs/{run_id}/events", response_model=None)
    def events(run_id: str, cursor: int = 0, limit: int = 100) -> dict[str, Any] | JSONResponse:
        try:
            values, next_cursor = controller.read_events(run_id, cursor=cursor, limit=limit)
            return {
                "events": [item.model_dump() for item in values],
                "cursor": cursor,
                "next_cursor": next_cursor,
            }
        except KeyError as exc:
            return _error(exc, code="RUN_NOT_FOUND", status=404)
        except ValueError as exc:
            return _error(exc, code="INVALID_CURSOR", status=422)
        except RuntimeError as exc:
            return _error(exc, code="EVENTS_UNAVAILABLE", status=503, retryable=True)

    @app.post("/runs/{run_id}/vertex-submit", response_model=None)
    def vertex_submit(run_id: str) -> dict[str, str] | JSONResponse:
        try:
            return {"vertex_job_name": controller.dispatch_vertex(run_id)}
        except KeyError as exc:
            return _error(exc, code="RUN_NOT_FOUND", status=404)
        except (ValueError, RuntimeError) as exc:
            return _error(exc, code="DISPATCH_CONFLICT", status=409)

    @app.post("/runs/{run_id}/events", response_model=None)
    def event(run_id: str, body: EventBody) -> dict[str, Any] | JSONResponse:
        try:
            if body.state is not None:
                controller.update_state(
                    run_id,
                    body.state,
                    vertex_job_name=body.vertex_job_name,
                    failure_code=body.failure_code,
                    failure_message=body.failure_message,
                    logs_uri=body.logs_uri,
                    metrics=body.metrics,
                )
            if body.event_type:
                controller.append_event(
                    run_id, body.event_type, body.payload or {}, event_id=body.event_id
                )
            elif body.state is None:
                raise ValueError("state or event_type is required")
            return controller.status(run_id, detail=True)
        except KeyError as exc:
            return _error(exc, code="RUN_NOT_FOUND", status=404)
        except ValueError as exc:
            return _error(exc, code="INVALID_EVENT", status=409)
        except RuntimeError as exc:
            return _error(exc, code="EVENTS_UNAVAILABLE", status=503, retryable=True)

    @app.post("/batches/plan", response_model=None)
    def plan_batch(body: BatchPlanBody) -> dict[str, Any] | JSONResponse:
        try:
            # Validate dimensions before invoking any cloud adapter.
            expand_matrix(body.matrix)
            plans = controller.plan_batch(
                experiment=body.experiment,
                job=body.job,
                matrix=body.matrix,
                idempotency_key=body.idempotency_key,
            )
            return {"plans": plans, "count": len(plans)}
        except ValueError as exc:
            return _error(exc, code="INVALID_BATCH", status=422)

    @app.post("/batches", status_code=202, response_model=None)
    def submit_batch(body: BatchBody) -> dict[str, Any] | JSONResponse:
        try:
            results = controller.submit_batch(
                experiment=body.experiment,
                job=body.job,
                matrix=body.matrix,
                idempotency_key=body.idempotency_key,
                runtime_override=body.runtime,
                dataset_override=body.dataset,
                classes=body.classes,
            )
            return {
                "runs": [run.model_dump(mode="json") for run, _ in results],
                "created": [created for _, created in results],
            }
        except ValueError as exc:
            return _error(exc, code="INVALID_BATCH", status=422)
        except RuntimeError as exc:
            return _error(exc, code="BATCH_SUBMISSION_FAILED", status=503, retryable=True)

    return app


class _InlineRuntimeCatalog:
    def get_certified(self, runtime_id: str) -> CertifiedRuntime | None:
        del runtime_id
        return None


class _InlineDatasetCatalog:
    def get(self, version_id: str) -> DatasetVersion | None:
        del version_id
        return None


def create_default_app():
    """Build the Cloud Run app from environment config without importing cloud SDKs."""
    import os

    tracking_uri = os.environ.get("DEFECT_MLFLOW_TRACKING_URI") or os.environ.get(
        "MLFLOW_TRACKING_URI"
    )
    tracker = MLflowTracker(tracking_uri, iam_auth=True) if tracking_uri else None
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "")
    workflow_name = os.environ.get("DEFECT_WORKFLOW_NAME", "")
    control_url = os.environ.get("CONTROL_URL", "")
    workflows = (
        GoogleWorkflowsStarter(
            project=project,
            region=region,
            workflow_name=workflow_name,
            control_service_url=control_url or None,
        )
        if workflow_name
        else None
    )
    configure_logging()
    intake_store = intake_store_from_env()
    controller = RunController(
        store=run_store_from_env(),
        runtimes=_InlineRuntimeCatalog(),
        datasets=_InlineDatasetCatalog(),
        catalogs=catalog_store_from_env(),
        capabilities=load_capabilities_from_env(),
        require_capabilities=True,
        vertex=VertexAiplatformSubmitter(),
        workflows=workflows,
        tracker=tracker,
    )
    small_model = os.environ.get("DEFECT_LITELLM_SMALL_MODEL")
    large_model = os.environ.get("DEFECT_LITELLM_LARGE_MODEL")
    routing = (
        ModelRoutingPolicy(small_model=small_model, large_model=large_model)
        if small_model and large_model
        else None
    )
    return create_app(
        controller,
        intake_coordinator=IntakeCoordinator(
            intake_store, model=os.environ.get("DEFECT_LITELLM_MODEL"), routing_policy=routing
        ),
        reference_store=reference_catalog_store_from_env(),
    )


class _LazyASGIApp:
    """Keep package import lightweight; initialize FastAPI only when serving starts."""

    def __init__(self):
        self._app = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            self._app = create_default_app()
        await self._app(scope, receive, send)


app = _LazyASGIApp()
